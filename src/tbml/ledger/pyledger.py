"""Pure-Python evidence ledger: a signed, hash-chained, append-only log.

Fallback backend for machines without Node.js / Hardhat, and the backend used
by the test suite. It offers the same guarantees as the contract in a
single-process form:

* **append-only and tamper-evident**: every entry stores the hash of the
  previous entry; changing or removing an entry breaks the chain
  (:meth:`PyLedger.verify_chain`);
* **permissioned**: entries are signed (secp256k1, the same curve Ethereum
  uses). The admin key registers organisations; an evidence entry is accepted
  only if it is signed by a registered organisation;
* **immutable records**: a hash can be registered once.

It is *not* a distributed ledger: there is one copy and no consensus. That is
the price of the fallback, stated in the limitations.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from eth_account import Account
from eth_account.messages import encode_defunct

from tbml.ledger.base import EvidenceLedger, EvidenceRecord, Journal, LedgerError, PermissionDenied, Receipt, normalise_digest
from tbml.ledger.hashing import canonical_json, sha256_hex

GENESIS_HASH = "0" * 64
ADMIN = "admin"


def _key_for(name: str, salt: str) -> Any:
    """Deterministic demo key for an organisation (real deployments hold their own keys)."""
    return Account.from_key(hashlib.sha256(f"tbml-pyledger|{salt}|{name}".encode()).digest())


class PyLedger(EvidenceLedger):
    backend = "python"

    def __init__(self, directory: str | Path, admin: str = ADMIN, key_salt: str = "demo") -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.chain_path = self.dir / "chain.jsonl"
        self.journal = Journal(self.dir / "journal.jsonl")
        self._salt = key_salt
        self._admin_name = admin
        self._keys: dict[str, Any] = {admin: _key_for(admin, key_salt)}
        self._orgs: dict[str, str] = {}  # active organisations: name -> address
        self._names: dict[str, str] = {}  # address -> name (ever registered)
        self._index: dict[str, dict[str, Any]] = {}
        self._entries: list[dict[str, Any]] = []
        self._bytes = 0
        self._sign_ops = 0
        self._load()

    # ------------------------------------------------------------------ chain mechanics
    @property
    def admin_address(self) -> str:
        return self._keys[self._admin_name].address

    def _load(self) -> None:
        if not self.chain_path.is_file():
            return
        with open(self.chain_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entry = json.loads(line)
                    self._apply(entry)
                    self._bytes += len(line.encode("utf-8"))

    def _apply(self, entry: dict[str, Any]) -> None:
        """Update the in-memory state with an entry that is already on the chain."""
        self._entries.append(entry)
        kind, payload = entry["kind"], entry["payload"]
        if kind == "org":
            self._orgs[payload["name"]] = payload["address"]
            self._names[payload["address"]] = payload["name"]
            # keys are derived from the name, so an organisation found on a reopened ledger can sign again
            key = _key_for(payload["name"], self._salt)
            if key.address == payload["address"]:
                self._keys.setdefault(payload["name"], key)
        elif kind == "revoke":
            self._orgs.pop(payload["name"], None)
        elif kind == "evidence":
            for item in payload["items"]:
                self._index[item["digest"]] = {**item, "doc_type": payload["doc_type"], "client_ts": payload["client_ts"],
                                               "submitter": entry["signer"], "block_ts": entry["ts"], "index": entry["index"]}

    @staticmethod
    def _entry_hash(entry: dict[str, Any]) -> str:
        return sha256_hex(canonical_json({k: v for k, v in entry.items() if k != "entry_hash"}))

    @staticmethod
    def _signing_digest(entry: dict[str, Any]) -> bytes:
        body = {k: v for k, v in entry.items() if k not in ("signature", "entry_hash")}
        return hashlib.sha256(canonical_json(body)).digest()

    def _append(self, kind: str, payload: dict[str, Any], signer_name: str) -> tuple[dict[str, Any], int]:
        key = self._keys.get(signer_name)
        if key is None:
            raise PermissionDenied(f"no signing key for '{signer_name}'")
        entry: dict[str, Any] = {
            "index": len(self._entries),
            "prev_hash": self._entries[-1]["entry_hash"] if self._entries else GENESIS_HASH,
            "ts": int(time.time()),
            "kind": kind,
            "signer": key.address,
            "payload": payload,
        }
        entry["signature"] = Account.sign_message(encode_defunct(self._signing_digest(entry)), key.key).signature.hex()
        entry["entry_hash"] = self._entry_hash(entry)
        self._sign_ops += 1
        line = json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n"
        with open(self.chain_path, "a", encoding="utf-8") as handle:
            handle.write(line)
        size = len(line.encode("utf-8"))
        self._bytes += size
        self._apply(entry)
        return entry, size

    # ------------------------------------------------------------------ organisations
    def register_organisation(self, name: str) -> str:
        if name in self._orgs:
            return self._orgs[name]
        self._keys.setdefault(name, _key_for(name, self._salt))
        address = self._keys[name].address
        self._append("org", {"name": name, "address": address}, self._admin_name)
        return address

    def revoke_organisation(self, name: str) -> None:
        if name in self._orgs:
            self._append("revoke", {"name": name}, self._admin_name)

    def organisations(self) -> dict[str, str]:
        return dict(self._orgs)

    # ------------------------------------------------------------------ writes
    def _check_writer(self, org: str) -> None:
        if org not in self._orgs:
            raise PermissionDenied(f"'{org}' is not a registered organisation")

    def register(self, digest: str, doc_type: str, ref_id: str, org: str, ts: int | None = None) -> Receipt:
        return self.register_batch([digest], doc_type, [ref_id], org, ts)[0]

    def register_batch(self, digests: list[str], doc_type: str, ref_ids: list[str], org: str, ts: int | None = None) -> list[Receipt]:
        if len(digests) != len(ref_ids):
            raise LedgerError("digests and ref_ids differ in length")
        self._check_writer(org)
        ts = self.now() if ts is None else int(ts)
        start = time.perf_counter()
        clean = [normalise_digest(d) for d in digests]
        fresh: list[tuple[str, str]] = []
        seen: set[str] = set()
        for d, ref in zip(clean, ref_ids):
            if d not in self._index and d not in seen:
                fresh.append((d, ref))
                seen.add(d)
        written = 0
        if fresh:
            _, written = self._append("evidence", {"doc_type": doc_type, "client_ts": ts, "items": [{"digest": d, "ref_id": r} for d, r in fresh]}, org)
        latency = time.perf_counter() - start
        fresh_set = {d for d, _ in fresh}
        receipts = []
        for d, ref in zip(clean, ref_ids):
            new = d in fresh_set
            fresh_set.discard(d)
            known = self._index[d]
            receipt = Receipt(
                digest=d, doc_type=doc_type, ref_id=ref, org=org, status="registered" if new else "duplicate",
                tx_id=f"py:{known['index']}", block=int(known["index"]), client_ts=ts,
                cost=int(written / len(fresh)) if new else 0, latency_s=latency / max(len(digests), 1),
            )
            self.journal.append(receipt)
            receipts.append(receipt)
        return receipts

    # ------------------------------------------------------------------ reads
    def verify(self, digest: str) -> EvidenceRecord:
        d = normalise_digest(digest)
        item = self._index.get(d)
        if item is None:
            return EvidenceRecord(digest=d, exists=False)
        return EvidenceRecord(digest=d, exists=True, org=self._names.get(item["submitter"], ""), submitter=item["submitter"],
                              doc_type=item["doc_type"], ref_id=item["ref_id"], client_ts=int(item["client_ts"]), block_ts=int(item["block_ts"]))

    def audit_trail(self, doc_type: str | None = None) -> list[dict[str, Any]]:
        trail = []
        for entry in self._entries:
            if entry["kind"] != "evidence" or (doc_type and entry["payload"]["doc_type"] != doc_type):
                continue
            for item in entry["payload"]["items"]:
                trail.append({"digest": item["digest"], "ref_id": item["ref_id"], "doc_type": entry["payload"]["doc_type"],
                              "org": self._names.get(entry["signer"], ""), "submitter": entry["signer"],
                              "client_ts": entry["payload"]["client_ts"], "block_ts": entry["ts"], "block": entry["index"], "tx_id": f"py:{entry['index']}"})
        return trail

    def stats(self) -> dict[str, Any]:
        return {"backend": self.backend, "n_records": len(self._index), "n_entries": len(self._entries), "n_organisations": len(self._orgs),
                "storage_bytes": self._bytes, "total_cost": self._bytes, "cost_unit": "bytes appended", "sign_ops": self._sign_ops}

    # ------------------------------------------------------------------ integrity of the ledger itself
    def verify_chain(self) -> tuple[bool, str]:
        """Re-read the chain file and check links, hashes, signatures and permissions."""
        if not self.chain_path.is_file():
            return True, "empty ledger"
        prev = GENESIS_HASH
        admin = self.admin_address
        orgs: set[str] = set()
        seen: set[str] = set()
        position = -1
        with open(self.chain_path, "r", encoding="utf-8") as handle:
            for position, line in enumerate(l for l in handle if l.strip()):
                entry = json.loads(line)
                if entry["index"] != position:
                    return False, f"entry {position}: index is {entry['index']} (an entry was removed or reordered)"
                if entry["prev_hash"] != prev:
                    return False, f"entry {position}: broken link to the previous entry"
                if self._entry_hash(entry) != entry["entry_hash"]:
                    return False, f"entry {position}: content does not match its hash"
                signer = Account.recover_message(encode_defunct(self._signing_digest(entry)), signature=bytes.fromhex(entry["signature"].removeprefix("0x")))
                if signer != entry["signer"]:
                    return False, f"entry {position}: signature does not match the stated signer"
                if entry["kind"] in ("org", "revoke"):
                    if signer != admin:
                        return False, f"entry {position}: organisation change not signed by the admin"
                    if entry["kind"] == "org":
                        orgs.add(entry["payload"]["address"])
                    else:
                        orgs = {a for a in orgs if self._names.get(a) != entry["payload"]["name"]}
                elif entry["kind"] == "evidence":
                    if signer not in orgs:
                        return False, f"entry {position}: evidence signed by an unregistered organisation"
                    for item in entry["payload"]["items"]:
                        if item["digest"] in seen:
                            return False, f"entry {position}: hash registered twice"
                        seen.add(item["digest"])
                prev = entry["entry_hash"]
        return True, f"{position + 1} entries verified"
