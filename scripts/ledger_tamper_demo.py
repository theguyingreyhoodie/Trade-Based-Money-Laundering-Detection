#!/usr/bin/env python
"""Phase 5b: integrity verification demo.

1. A stored invoice is verified against the ledger (passes), then one field is
   changed and it is verified again (fails). Changing the off-chain index as
   well does not help the attacker.
2. The global model saved by the primary federated run is verified against the
   hash registered for its round (passes); one weight is changed (fails).
3. If alerts have been generated, one alert is verified and then altered.

Writes ``results/ledger/tamper_demo.json``. Exits with an error if any step
does not behave as expected.
"""

from __future__ import annotations

import copy

import numpy as np

from tbml.data.generator import load_dataset
from tbml.fl.experiments import primary_scenario
from tbml.fl.simulation import model_dir
from tbml.ledger.audit import verify_model
from tbml.ledger.evidence import DOCUMENT_TABLES, EvidenceStore, store_path, verify_document
from tbml.ledger.factory import open_ledger
from tbml.ledger.hashing import hash_record
from tbml.utils import get_logger, load_json, parse_cli, save_json

log = get_logger("ledger_tamper_demo")


def document_demo(ledger, store, ds) -> dict:
    doc_type, id_col, _ = DOCUMENT_TABLES["invoices"]
    invoices = ds.tables["invoices"].drop_duplicates(id_col)
    record = invoices.iloc[len(invoices) // 2].to_dict()
    ref = str(record[id_col])
    original = verify_document(ledger, store, doc_type, ref, record)

    tampered_record = dict(record)
    tampered_record["invoice_value"] = round(float(record["invoice_value"]) * 1.10, 2)
    tampered = verify_document(ledger, store, doc_type, ref, tampered_record)

    # a stronger attacker also rewrites the off-chain index to match the altered document
    store.set_hash(doc_type, ref, hash_record(tampered_record))
    index_rewritten = verify_document(ledger, store, doc_type, ref, tampered_record)
    store.set_hash(doc_type, ref, original["computed_hash"])
    restored = verify_document(ledger, store, doc_type, ref, record)

    return {"document": ref, "field_changed": "invoice_value", "original_value": record["invoice_value"], "tampered_value": tampered_record["invoice_value"],
            "original": original, "tampered": tampered, "tampered_with_index_rewritten": index_rewritten, "restored": restored,
            "as_expected": original["verified"] and not tampered["verified"] and not index_rewritten["verified"] and restored["verified"]}


def model_demo(ledger, cfg) -> dict | None:
    seed = int(cfg.project.seed)
    mdir = model_dir(cfg, seed, primary_scenario(cfg), str(cfg.fl.primary.strategy), "primary")
    path = mdir / "global_model.npz"
    if not path.is_file():
        log.warning("no primary global model found (run scripts/run_fl.py first): skipping the model check")
        return None
    meta = load_json(mdir / "model_meta.json")
    with np.load(path) as z:
        names = list(z.files)
        weights = [z[n] for n in names]
    digest, record = verify_model(ledger, names, weights)
    altered = copy.deepcopy(weights)
    altered[2].flat[0] += np.float32(1e-3)  # nudge a single weight of the first layer
    digest_altered, record_altered = verify_model(ledger, names, altered)
    shown = path.relative_to(cfg.root) if path.is_relative_to(cfg.root) else path
    return {"model_file": str(shown), "round": meta["best_round"],
            "original": {"sha256": digest, "on_ledger": record.exists, "doc_type": record.doc_type, "ref_id": record.ref_id, "registered_by": record.org},
            "one_weight_changed": {"sha256": digest_altered, "on_ledger": record_altered.exists},
            "as_expected": record.exists and record.ref_id.endswith(f"round-{int(meta['best_round']):03d}") and not record_altered.exists}


def alert_demo(ledger, cfg) -> dict | None:
    path = cfg.path("results_dir", "alerts", "alerts.json")
    if not path.is_file():
        return None
    from tbml.explain.alerts import alert_digest

    alerts = load_json(path)["alerts"]
    if not alerts:
        return None
    alert = alerts[0]
    genuine = ledger.verify(alert_digest(alert))
    forged = dict(alert)
    forged["risk_score"] = max(0.0, float(alert["risk_score"]) - 40.0)  # someone lowers the score after the fact
    forged_record = ledger.verify(alert_digest(forged))
    return {"alert_id": alert["alert_id"], "original": {"on_ledger": genuine.exists, "registered_by": genuine.org, "ref_id": genuine.ref_id},
            "score_lowered": {"on_ledger": forged_record.exists}, "as_expected": genuine.exists and not forged_record.exists}


def main() -> None:
    _, cfg = parse_cli(__doc__)
    ds = load_dataset(cfg.seed_dir("data_dir", int(cfg.project.seed)))
    ledger = open_ledger(cfg)
    store = EvidenceStore(store_path(cfg))
    report = {"backend": ledger.backend, "trade_document": document_demo(ledger, store, ds), "model_update": model_demo(ledger, cfg), "alert": alert_demo(ledger, cfg)}
    store.close()
    ledger.close()
    save_json(report, cfg.path("results_dir", "ledger", "tamper_demo.json"))
    failed = [name for name, part in report.items() if isinstance(part, dict) and part.get("as_expected") is False]
    for name in ("trade_document", "model_update", "alert"):
        part = report[name]
        log.info("%-15s %s", name, "not run" if part is None else ("genuine verified, tampered rejected" if part["as_expected"] else "UNEXPECTED RESULT"))
    if failed:
        raise SystemExit(f"tamper demo failed for: {failed}")


if __name__ == "__main__":
    main()
