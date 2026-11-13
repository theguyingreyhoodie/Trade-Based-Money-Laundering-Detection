"""Evidence layer: hashing, Merkle batching, both ledger backends, tamper detection."""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import subprocess
import time

import numpy as np
import pandas as pd
import pytest

from tbml.ledger.audit import GLOBAL_MODEL, MODEL_UPDATE, ModelAuditSink, verify_model
from tbml.ledger.base import EvidenceLedger, LedgerError, PermissionDenied, normalise_digest, storage_slots
from tbml.ledger.bench import run_benchmark
from tbml.ledger.evidence import BATCH_PREFIX, DOCUMENT_TABLES, EvidenceStore, document_hashes, register_documents, verify_document
from tbml.ledger.factory import open_ledger, resolve_backend
from tbml.ledger.hashing import canonical_json, hash_arrays, hash_file, hash_record, normalise
from tbml.ledger.merkle import merkle_proof, merkle_root, root_from_proof, verify_proof
from tbml.ledger.node import artifact_path, chain_dir, hardhat_installed, rpc_alive
from tbml.ledger.pyledger import PyLedger

TEST_RPC = "http://127.0.0.1:8546"  # separate from the development chain on 8545


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --------------------------------------------------------------------------- canonical hashing


def test_canonical_json_is_order_and_format_independent() -> None:
    a = {"invoice_id": "INV-1", "value": 1200.0, "qty": 3, "date": pd.Timestamp("2024-03-05"), "note": None}
    b = {"note": float("nan"), "date": dt.date(2024, 3, 5), "qty": np.int64(3), "value": 1200, "invoice_id": " INV-1 "}
    assert canonical_json(a) == canonical_json(b)
    assert hash_record(a) == hash_record(b)
    assert json.loads(canonical_json(a)) == {"date": "2024-03-05", "invoice_id": "INV-1", "note": None, "qty": 3, "value": 1200}
    assert normalise(np.float32(0.1)) == 0.1 and normalise(1e-9) == 0 and normalise(2.5) == 2.5
    assert normalise(pd.Timestamp("2024-03-05 13:45")) == "2024-03-05T13:45:00"
    assert b" " not in canonical_json({"a": [1, 2], "b": {"c": "x y"}}).replace(b"x y", b"")


@pytest.mark.parametrize("field,value", [("value", 1200.01), ("qty", 4), ("invoice_id", "INV-2"), ("date", pd.Timestamp("2024-03-06")), ("note", "x")])
def test_any_field_change_changes_the_hash(field, value) -> None:
    record = {"invoice_id": "INV-1", "value": 1200.0, "qty": 3, "date": pd.Timestamp("2024-03-05"), "note": None}
    changed = dict(record)
    changed[field] = value
    assert hash_record(changed) != hash_record(record)
    assert len(hash_record(record)) == 64


def test_file_and_array_hashes(tmp_path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"evidence")
    assert hash_file(path) == hashlib.sha256(b"evidence").hexdigest()
    w = [np.arange(6, dtype=np.float32).reshape(2, 3), np.ones(2, dtype=np.float32)]
    assert hash_arrays(zip(["a", "b"], w)) == hash_arrays(zip(["a", "b"], [x.copy() for x in w]))
    assert hash_arrays(zip(["a", "b"], w)) != hash_arrays(zip(["a", "b"], [w[0].reshape(3, 2), w[1]])), "shape is part of the hash"
    assert hash_arrays(zip(["a", "b"], w)) != hash_arrays(zip(["a", "b"], [w[0].astype(np.float64), w[1]])), "dtype is part of the hash"


def test_digest_validation() -> None:
    d = digest("x")
    assert normalise_digest("0x" + d.upper()) == d
    assert normalise_digest(bytes.fromhex(d)) == d
    for bad in ["abc", "g" * 64, "0" * 64, d + "00"]:
        with pytest.raises(LedgerError):
            normalise_digest(bad)
    assert storage_slots("invoice", "INV-0000001") == 3
    assert storage_slots("model_update", "seed0-n3_dir0.5-fedavg/round-001/client-0") == 5


# --------------------------------------------------------------------------- Merkle trees


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 13, 100])
def test_merkle_proofs(n: int) -> None:
    leaves = [digest(f"doc{i}") for i in range(n)]
    root = merkle_root(leaves)
    for i in range(n):
        proof = merkle_proof(leaves, i)
        assert verify_proof(leaves[i], proof, root)
        assert len(proof) <= int(np.ceil(np.log2(max(n, 2))))
        assert not verify_proof(digest("forged"), proof, root)
    if n > 1:
        assert not verify_proof(leaves[0], merkle_proof(leaves, 1), root), "a proof belongs to one position"
        assert merkle_root(leaves[::-1]) != root, "order matters"
        assert merkle_root(leaves[:-1]) != root


def test_merkle_domain_separation() -> None:
    """An inner node cannot be presented as a document hash."""
    leaves = [digest(f"doc{i}") for i in range(4)]
    root = merkle_root(leaves)
    left = hashlib.sha256(b"\x01" + hashlib.sha256(b"\x00" + bytes.fromhex(leaves[0])).digest() + hashlib.sha256(b"\x00" + bytes.fromhex(leaves[1])).digest()).hexdigest()
    assert merkle_root([left, *leaves[2:]]) != root
    assert root_from_proof(leaves[0], merkle_proof(leaves, 0)) == root
    with pytest.raises(ValueError):
        merkle_root([])
    with pytest.raises(IndexError):
        merkle_proof(leaves, 4)


# --------------------------------------------------------------------------- backends


@pytest.fixture(scope="session")
def evm_node(cfg):
    """A throw-away Hardhat node for the tests (skipped when Node.js / Hardhat is not installed)."""
    if not hardhat_installed(cfg) or not artifact_path(cfg).is_file():
        pytest.skip("Hardhat is not installed (run `make chain-install`); EVM tests skipped")
    started = None
    if not rpc_alive(TEST_RPC):
        started = subprocess.Popen([str(chain_dir(cfg) / "node_modules" / ".bin" / "hardhat"), "node", "--hostname", "127.0.0.1", "--port", "8546"],
                                   cwd=chain_dir(cfg), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 40
        while not rpc_alive(TEST_RPC):
            if time.monotonic() > deadline or started.poll() is not None:
                started.kill()
                pytest.skip("could not start a Hardhat node for the tests")
            time.sleep(0.3)
    yield TEST_RPC
    if started is not None:
        started.terminate()
        started.wait(timeout=10)


@pytest.fixture(params=["python", "evm"])
def ledger(request, cfg, tmp_path) -> EvidenceLedger:
    """A fresh, empty ledger of each backend with two organisations registered."""
    config = copy.deepcopy(cfg)
    config.set_path("ledger.auto_start_node", False)
    if request.param == "evm":
        config.set_path("ledger.rpc_url", request.getfixturevalue("evm_node"))
    led = open_ledger(config, backend=request.param, fresh=True, state_dir=tmp_path / "ledger")
    yield led
    led.close()


def test_register_and_verify(ledger) -> None:
    assert {"bank_consortium", "trade_registry", "customs_authority", "fl_aggregator"} <= set(ledger.organisations())
    d = digest("invoice-1")
    before = int(time.time()) - 5
    receipt = ledger.register(d, "invoice", "INV-0000001", "trade_registry", ts=1_700_000_000)
    assert receipt.status == "registered" and receipt.digest == d and receipt.cost > 0 and receipt.tx_id
    record = ledger.verify(d)
    assert record.exists and record.doc_type == "invoice" and record.ref_id == "INV-0000001"
    assert record.org == "trade_registry" and record.submitter == ledger.organisations()["trade_registry"]
    assert record.client_ts == 1_700_000_000 and record.block_ts >= before
    assert ledger.is_registered("0x" + d.upper())
    unknown = ledger.verify(digest("never registered"))
    assert not unknown.exists and unknown.org == "" and unknown.ref_id == ""
    assert ledger.stats()["n_records"] == 1


def test_only_registered_organisations_can_write(ledger) -> None:
    """The permissioned part: an organisation that is not on the allowlist is refused."""
    with pytest.raises(PermissionDenied):
        ledger.register(digest("x"), "invoice", "X", "unknown_org")
    with pytest.raises(PermissionDenied):
        ledger.register_batch([digest("y")], "invoice", ["Y"], "unknown_org")
    assert not ledger.is_registered(digest("x")) and ledger.stats()["n_records"] == 0

    ledger.register_organisation("new_bank")
    assert ledger.register(digest("x"), "payment", "PAY-1", "new_bank").status == "registered"
    ledger.revoke_organisation("new_bank")
    assert "new_bank" not in ledger.organisations()
    with pytest.raises(PermissionDenied):
        ledger.register(digest("z"), "payment", "PAY-2", "new_bank")
    assert ledger.verify(digest("x")).exists, "revoking an organisation does not erase what it registered"


def test_records_are_immutable(ledger) -> None:
    d = digest("doc")
    first = ledger.register(d, "invoice", "INV-1", "trade_registry", ts=100)
    again = ledger.register(d, "payment", "OVERWRITE", "bank_consortium", ts=999)
    assert first.status == "registered" and again.status == "duplicate"
    record = ledger.verify(d)
    assert (record.doc_type, record.ref_id, record.org, record.client_ts) == ("invoice", "INV-1", "trade_registry", 100)
    assert ledger.stats()["n_records"] == 1
    with pytest.raises(LedgerError):
        ledger.register("0" * 64, "invoice", "ZERO", "trade_registry")


def test_batch_registration_and_audit_trail(ledger) -> None:
    digests = [digest(f"d{i}") for i in range(250)]
    refs = [f"REF-{i:04d}" for i in range(250)]
    ledger.register(digests[7], "invoice", refs[7], "trade_registry")
    receipts = ledger.register_batch(digests, "invoice", refs, "trade_registry", ts=1234)
    assert len(receipts) == 250
    assert [r.status for r in receipts].count("duplicate") == 1 and receipts[7].status == "duplicate"
    assert all(ledger.verify(d).exists for d in digests[::25])
    assert ledger.stats()["n_records"] == 250
    single_cost = ledger.register(digest("single"), "invoice", "S", "trade_registry").cost
    batch_cost = np.mean([r.cost for r in receipts if r.status == "registered"])
    assert batch_cost < single_cost, "batching must be cheaper per record"

    trail = ledger.audit_trail("invoice")
    assert len(trail) == 251
    assert {t["org"] for t in trail} == {"trade_registry"}
    assert {t["digest"] for t in trail} == set(digests) | {digest("single")}
    assert trail[0]["ref_id"] == refs[7] and trail[0]["block"] <= trail[-1]["block"]
    assert ledger.audit_trail("payment") == []
    with pytest.raises(LedgerError):
        ledger.register_batch(digests[:2], "invoice", refs[:1], "trade_registry")
    # the off-chain journal mirrors what was registered (duplicates are not journalled)
    assert len(ledger.journal.entries()) == 251


def test_storage_grows_linearly(ledger) -> None:
    sizes = []
    for step in range(4):
        ledger.register_batch([digest(f"s{step}-{i}") for i in range(40)], "invoice", [f"S{step}-{i:03d}" for i in range(40)], "trade_registry")
        sizes.append(ledger.stats()["storage_bytes"])
    growth = np.diff(sizes)
    assert (growth > 0).all() and growth.max() - growth.min() <= 0.05 * growth.mean()
    assert ledger.stats()["total_cost"] > 0


# --------------------------------------------------------------------------- pure-Python ledger: tamper evidence of the ledger itself


def test_pyledger_persists_and_detects_tampering(tmp_path) -> None:
    led = PyLedger(tmp_path / "led")
    led.register_organisation("bank")
    led.register_batch([digest(f"p{i}") for i in range(5)], "payment", [f"PAY-{i}" for i in range(5)], "bank")
    led.register(digest("solo"), "invoice", "INV-9", "bank")
    assert led.verify_chain()[0]

    reopened = PyLedger(tmp_path / "led")
    assert reopened.verify(digest("p3")).ref_id == "PAY-3" and reopened.stats()["n_records"] == 6
    assert reopened.organisations() == led.organisations()

    path = tmp_path / "led" / "chain.jsonl"
    original = path.read_text()
    lines = original.splitlines()

    edited = json.loads(lines[1])
    edited["payload"]["items"][0]["ref_id"] = "PAY-FORGED"  # rewrite history
    path.write_text("\n".join([lines[0], json.dumps(edited), *lines[2:]]) + "\n")
    ok, reason = PyLedger(tmp_path / "led").verify_chain()
    assert not ok and "entry 1" in reason

    path.write_text("\n".join([lines[0], lines[2]]) + "\n")  # delete an entry
    assert not PyLedger(tmp_path / "led").verify_chain()[0]

    forged = json.loads(lines[2])
    forged["signer"] = "0x" + "11" * 20  # claim another author
    forged["entry_hash"] = PyLedger._entry_hash(forged)
    path.write_text("\n".join([lines[0], lines[1], json.dumps(forged)]) + "\n")
    assert not PyLedger(tmp_path / "led").verify_chain()[0]

    path.write_text(original)
    assert PyLedger(tmp_path / "led").verify_chain()[0]


def test_backend_selection(cfg, tmp_path) -> None:
    config = copy.deepcopy(cfg)
    assert resolve_backend(config, "python") == "python" and resolve_backend(config, "evm") == "evm"
    config.set_path("ledger.backend", "auto")
    config.set_path("ledger.rpc_url", "http://127.0.0.1:59999")
    config.set_path("ledger.auto_start_node", False)
    assert resolve_backend(config) == "python", "without a node the pipeline falls back to the pure-Python ledger"
    with pytest.raises(ValueError):
        resolve_backend(config, "fabric")
    led = open_ledger(config, state_dir=tmp_path / "auto")
    assert led.backend == "python" and set(config.ledger.organisations) <= set(led.organisations())
    led.close()


# --------------------------------------------------------------------------- trade documents


@pytest.fixture
def registered(ledger, cfg, dataset, tmp_path):
    store = EvidenceStore(tmp_path / "evidence.sqlite")
    summary = register_documents(ledger, store, dataset.tables, cfg, run_id="t", limit=1200)
    yield ledger, store, summary
    store.close()


def test_documents_are_anchored_in_batches(registered, dataset, cfg) -> None:
    ledger, store, summary = registered
    assert summary["documents"] == sum(t["documents"] for t in summary["tables"].values()) == store.counts()["documents"]
    assert summary["batches"] == store.counts()["batches"] == ledger.stats()["n_records"], "one ledger record per batch, not per document"
    assert summary["batches"] < summary["documents"] / 100
    owners = {t["doc_type"]: t["org"] for t in summary["tables"].values()}
    assert owners == {"invoice": "trade_registry", "shipment": "trade_registry", "payment": "bank_consortium", "customs_declaration": "customs_authority"}
    for entry in ledger.audit_trail():
        assert entry["doc_type"].startswith(BATCH_PREFIX)
        assert entry["org"] == owners[entry["doc_type"].removeprefix(BATCH_PREFIX)]
    # nothing but hashes and opaque batch ids is on the ledger
    known_ids = set(dataset.tables["invoices"]["invoice_id"]) | set(dataset.tables["payments"]["payer_id"])
    assert not any(entry["ref_id"] in known_ids for entry in ledger.audit_trail())


@pytest.mark.parametrize("table", list(DOCUMENT_TABLES))
def test_genuine_documents_verify(registered, dataset, table) -> None:
    ledger, store, _ = registered
    doc_type, id_col, org = DOCUMENT_TABLES[table]
    docs = dataset.tables[table].drop_duplicates(id_col).head(1200)
    for record in docs.sample(5, random_state=1).to_dict("records"):
        result = verify_document(ledger, store, doc_type, str(record[id_col]), record)
        assert result["verified"] and result["hash_matches_index"] and result["root_on_ledger"], result
        assert result["registered_by"] == org and result["proof_length"] <= 9


def test_tampered_invoice_fails_verification(registered, dataset) -> None:
    """The verification demo: change one field of a stored invoice and verification fails."""
    ledger, store, _ = registered
    record = dataset.tables["invoices"].drop_duplicates("invoice_id").iloc[10].to_dict()
    ref = record["invoice_id"]
    assert verify_document(ledger, store, "invoice", ref, record)["verified"]

    for field, value in [("invoice_value", round(record["invoice_value"] * 1.1, 2)), ("quantity", record["quantity"] + 1), ("importer_id", "ENT-99999"),
                         ("hs_code", "710812"), ("invoice_date", record["invoice_date"] + pd.Timedelta(days=1))]:
        tampered = dict(record)
        tampered[field] = value
        result = verify_document(ledger, store, "invoice", ref, tampered)
        assert not result["verified"] and not result["hash_matches_index"], field
        assert "differs" in result["reason"]

    # an attacker who also rewrites the off-chain index still fails: the ledger holds the batch root
    forged = dict(record)
    forged["invoice_value"] = 1.0
    store.set_hash("invoice", ref, hash_record(forged))
    result = verify_document(ledger, store, "invoice", ref, forged)
    assert result["hash_matches_index"] and not result["root_on_ledger"] and not result["verified"]
    assert "index" in result["reason"]
    # the genuine document still verifies: its own hash is recomputed, never read from the index
    genuine = verify_document(ledger, store, "invoice", ref, record)
    assert genuine["verified"] and not genuine["hash_matches_index"]
    # corrupting a neighbour's entry in the index can block verification, but can never make a forgery pass
    neighbour = dataset.tables["invoices"].drop_duplicates("invoice_id").iloc[11]["invoice_id"]
    store.set_hash("invoice", neighbour, hash_record({"anything": 1}))
    blocked = verify_document(ledger, store, "invoice", ref, record)
    assert not blocked["verified"] and not blocked["root_on_ledger"]
    assert not verify_document(ledger, store, "invoice", ref, forged)["verified"]

    assert not verify_document(ledger, store, "invoice", "INV-DOES-NOT-EXIST", record)["verified"]
    # a document presented under another document's reference fails as well
    other = dataset.tables["invoices"].drop_duplicates("invoice_id").iloc[11]["invoice_id"]
    assert not verify_document(ledger, store, "invoice", other, record)["verified"]


def test_document_hashes_are_per_document(dataset) -> None:
    hashes = document_hashes(dataset.tables["invoices"], "invoice_id")
    assert hashes["ref_id"].is_unique and hashes["sha256"].is_unique
    assert len(hashes) == dataset.tables["invoices"]["invoice_id"].nunique()


# --------------------------------------------------------------------------- model updates


def test_tampered_model_update_fails_verification(ledger) -> None:
    """Model hashes are registered per round; a changed weight no longer matches any record."""
    rng = np.random.default_rng(0)
    names = ["scaler.mean", "scaler.std", "body.0.weight", "body.0.bias"]
    make = lambda: [rng.normal(size=s).astype(np.float32) for s in [(5,), (5,), (4, 5), (4,)]]  # noqa: E731
    sink = ModelAuditSink(ledger, run_id="run-1")
    updates = {}
    for server_round in (1, 2):
        for client in range(3):
            updates[(server_round, client)] = make()
            sink({"round": server_round, "kind": "client_update", "client": client, "sha256": hash_arrays(zip(names, updates[(server_round, client)]))})
        updates[(server_round, None)] = make()
        entry = {"round": server_round, "kind": "global_model", "client": None, "sha256": hash_arrays(zip(names, updates[(server_round, None)]))}
        sink(entry)
        assert entry["ledger_status"] == "registered" and entry["tx_id"] and entry["ledger_org"] == "fl_aggregator"

    summary = sink.summary()
    assert summary["registered"] == 8 and summary["client_updates"] == 6 and summary["global_models"] == 2 and summary["cost_total"] > 0

    digest_, record = verify_model(ledger, names, updates[(2, 1)])
    assert record.exists and record.doc_type == MODEL_UPDATE and record.ref_id == "run-1/round-002/client-1" and record.org == "fl_client_1"
    _, record = verify_model(ledger, names, updates[(1, None)])
    assert record.exists and record.doc_type == GLOBAL_MODEL and record.ref_id == "run-1/round-001"

    tampered = [w.copy() for w in updates[(2, 1)]]
    tampered[2][0, 0] += np.float32(1e-4)  # a poisoned or swapped update
    assert not verify_model(ledger, names, tampered)[1].exists
    assert not verify_model(ledger, names, updates[(2, 1)][::-1])[1].exists
    assert {t["doc_type"] for t in ledger.audit_trail()} == {MODEL_UPDATE, GLOBAL_MODEL}


# --------------------------------------------------------------------------- restart and benchmark


def test_journal_replay_restores_evidence(cfg, ledger, tmp_path) -> None:
    """A development chain loses its state on restart; the journal re-anchors the evidence."""
    for i in range(6):
        ledger.register(digest(f"j{i}"), "alert", f"ALERT-{i}", "fl_aggregator", ts=1000 + i)
    entries = ledger.journal.entries()
    assert len(entries) == 6

    config = copy.deepcopy(cfg)
    config.set_path("ledger.auto_start_node", False)
    if ledger.backend == "evm":
        config.set_path("ledger.rpc_url", TEST_RPC)
    empty = open_ledger(config, backend=ledger.backend, fresh=True, state_dir=tmp_path / "second")
    assert not empty.is_registered(digest("j0"))
    assert empty.replay(entries) == 6
    record = empty.verify(digest("j4"))
    assert record.exists and record.ref_id == "ALERT-4" and record.client_ts == 1004 and record.org == "fl_aggregator"
    empty.close()


def test_benchmark_report(ledger, cfg) -> None:
    config = copy.deepcopy(cfg)
    config.set_path("ledger.benchmark", {"n_single": 15, "batch_sizes": [1, 10], "n_batched_items": 40, "block_interval_s": 0.25})
    report = run_benchmark(ledger, config)
    assert report["single"]["latency"]["p50_ms"] <= report["single"]["latency"]["p95_ms"]
    by_size = {row["batch_size"]: row for row in report["batched"]}
    assert by_size[10]["cost_per_record"] < by_size[1]["cost_per_record"]
    assert report["merkle"]["cost_per_record"] < by_size[10]["cost_per_record"] / 5, "a Merkle root amortises the cost over the whole batch"
    assert report["storage_bytes_per_record"] > 0 and len(report["storage_growth"]) >= 3
    if ledger.backend == "evm":
        interval = report["block_interval"]
        assert interval["sequential_latency"]["p50_ms"] > 100, "with timed blocks a registration waits for the next block"
        assert interval["concurrent"]["blocks_used"] <= 3
