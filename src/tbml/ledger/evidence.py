"""Registering and verifying trade documents.

At ingestion every raw document (invoice, shipment, payment, customs
declaration) is hashed. Hashes are grouped into batches; the Merkle root of each
batch is registered on the ledger by the organisation that owns the documents.
Raw documents stay with their owner.

An off-chain index (SQLite) remembers which batch and position each document
went into, so a proof can be rebuilt. The index is a convenience, not a source
of trust: verification recomputes the document hash, rebuilds the root through
the proof and asks the ledger whether *that root* was registered. Changing a
document, or the index, or both, yields a root the ledger has never seen.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from tbml.config import Config
from tbml.ledger.base import EvidenceLedger
from tbml.ledger.hashing import hash_record
from tbml.ledger.merkle import merkle_proof, merkle_root, root_from_proof
from tbml.utils import get_logger

log = get_logger(__name__)

#: raw table -> (document type, id column, owning organisation)
DOCUMENT_TABLES: dict[str, tuple[str, str, str]] = {
    "invoices": ("invoice", "invoice_id", "trade_registry"),
    "shipments": ("shipment", "bl_id", "trade_registry"),
    "payments": ("payment", "payment_id", "bank_consortium"),
    "declarations": ("customs_declaration", "declaration_id", "customs_authority"),
}
DOC_TYPE_TABLE = {doc_type: table for table, (doc_type, _, _) in DOCUMENT_TABLES.items()}
BATCH_PREFIX = "doc_batch:"


def document_hashes(table: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """One row per document: its reference id and the SHA-256 of its canonical JSON."""
    docs = table.drop_duplicates(id_col)
    return pd.DataFrame({"ref_id": docs[id_col].astype(str).to_numpy(), "sha256": [hash_record(r) for r in docs.to_dict("records")]})


class EvidenceStore:
    """Off-chain index: which batch and leaf position each document was registered in."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (doc_type TEXT, ref_id TEXT, sha256 TEXT, batch_id TEXT, leaf_index INTEGER,
                                                  PRIMARY KEY (doc_type, ref_id));
            CREATE INDEX IF NOT EXISTS documents_batch ON documents (batch_id, leaf_index);
            CREATE TABLE IF NOT EXISTS batches (batch_id TEXT PRIMARY KEY, doc_type TEXT, org TEXT, root TEXT, n_leaves INTEGER,
                                                tx_id TEXT, block INTEGER, client_ts INTEGER);
            """
        )

    def clear(self) -> None:
        self.db.executescript("DELETE FROM documents; DELETE FROM batches;")
        self.db.commit()

    def add_batch(self, batch_id: str, doc_type: str, org: str, root: str, ref_ids: list[str], hashes: list[str], tx_id: str, block: int, client_ts: int) -> None:
        self.db.execute("INSERT OR REPLACE INTO batches VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (batch_id, doc_type, org, root, len(hashes), tx_id, block, client_ts))
        self.db.executemany("INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?)", [(doc_type, r, h, batch_id, i) for i, (r, h) in enumerate(zip(ref_ids, hashes))])

    def commit(self) -> None:
        self.db.commit()

    def document(self, doc_type: str, ref_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT sha256, batch_id, leaf_index FROM documents WHERE doc_type = ? AND ref_id = ?", (doc_type, ref_id)).fetchone()
        return None if row is None else {"sha256": row[0], "batch_id": row[1], "leaf_index": int(row[2])}

    def batch(self, batch_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT doc_type, org, root, n_leaves, tx_id, block, client_ts FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
        return None if row is None else dict(zip(["doc_type", "org", "root", "n_leaves", "tx_id", "block", "client_ts"], row))

    def leaves(self, batch_id: str) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT sha256 FROM documents WHERE batch_id = ? ORDER BY leaf_index", (batch_id,))]

    def counts(self) -> dict[str, int]:
        docs = dict(self.db.execute("SELECT doc_type, COUNT(*) FROM documents GROUP BY doc_type").fetchall())
        return {"documents": int(sum(docs.values())), "batches": int(self.db.execute("SELECT COUNT(*) FROM batches").fetchone()[0]), **{f"n_{k}": int(v) for k, v in docs.items()}}

    def set_hash(self, doc_type: str, ref_id: str, sha256: str) -> None:
        """Overwrite an indexed hash. Exists only so tests and the demo can play the attacker."""
        self.db.execute("UPDATE documents SET sha256 = ? WHERE doc_type = ? AND ref_id = ?", (sha256, doc_type, ref_id))
        self.db.commit()

    def close(self) -> None:
        self.db.commit()
        self.db.close()


def store_path(cfg: Config) -> Path:
    return cfg.path("artifacts_dir", "ledger", "evidence.sqlite", create=True)


def register_documents(ledger: EvidenceLedger, store: EvidenceStore, tables: Mapping[str, pd.DataFrame], cfg: Config, run_id: str = "",
                       limit: int | None = None) -> dict[str, Any]:
    """Hash every trade document and anchor the hashes on the ledger as Merkle batches.

    Returns a summary with counts, timings and cost. ``limit`` caps the number
    of documents per table (used by tests and the benchmark).
    """
    batch_size = int(cfg.ledger.doc_batch_size)
    summary: dict[str, Any] = {"tables": {}, "batch_size": batch_size}
    total_docs = total_batches = total_cost = 0
    t_hash = t_ledger = 0.0
    for table, (doc_type, id_col, org) in DOCUMENT_TABLES.items():
        start = time.perf_counter()
        hashes = document_hashes(tables[table] if limit is None else tables[table].head(limit), id_col)
        t_hash += time.perf_counter() - start
        refs, digests = hashes["ref_id"].tolist(), hashes["sha256"].tolist()
        cost = n_batches = 0
        start = time.perf_counter()
        for b, offset in enumerate(range(0, len(digests), batch_size)):
            chunk_refs, chunk = refs[offset : offset + batch_size], digests[offset : offset + batch_size]
            root = merkle_root(chunk)
            batch_id = f"{doc_type}{('-' + run_id) if run_id else ''}-{b:05d}"
            receipt = ledger.register(root, BATCH_PREFIX + doc_type, batch_id, org)
            store.add_batch(batch_id, doc_type, org, root, chunk_refs, chunk, receipt.tx_id, receipt.block, receipt.client_ts)
            cost += receipt.cost
            n_batches += 1
        store.commit()
        t_ledger += time.perf_counter() - start
        summary["tables"][table] = {"doc_type": doc_type, "org": org, "documents": len(digests), "batches": n_batches, "cost": cost}
        total_docs += len(digests)
        total_batches += n_batches
        total_cost += cost
    stats = ledger.stats()
    summary.update({
        "backend": ledger.backend,
        "documents": total_docs,
        "batches": total_batches,
        "hash_seconds": t_hash,
        "ledger_seconds": t_ledger,
        "documents_per_second": total_docs / max(t_hash + t_ledger, 1e-9),
        "cost": total_cost,
        "cost_unit": stats["cost_unit"],
        "cost_per_document": total_cost / max(total_docs, 1),
        "ledger_storage_bytes": stats["storage_bytes"],
        "ledger_bytes_per_document": stats["storage_bytes"] / max(total_docs, 1),
    })
    log.info("%d documents anchored in %d batches on the %s ledger (%.1fs hashing, %.1fs ledger, %.0f %s per document)", total_docs, total_batches,
             ledger.backend, t_hash, t_ledger, summary["cost_per_document"], stats["cost_unit"])
    return summary


def verify_document(ledger: EvidenceLedger, store: EvidenceStore, doc_type: str, ref_id: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """Check a document as presented now against what was registered at ingestion."""
    computed = hash_record(record)
    result: dict[str, Any] = {"doc_type": doc_type, "ref_id": ref_id, "computed_hash": computed, "verified": False}
    indexed = store.document(doc_type, ref_id)
    if indexed is None:
        result["reason"] = "this document was never registered"
        return result
    leaves = store.leaves(indexed["batch_id"])
    proof = merkle_proof(leaves, indexed["leaf_index"])
    root = root_from_proof(computed, proof)
    on_ledger = ledger.verify(root)
    genuine = on_ledger.exists and on_ledger.doc_type == BATCH_PREFIX + doc_type
    result.update({
        "indexed_hash": indexed["sha256"],
        "hash_matches_index": computed == indexed["sha256"],
        "batch_id": indexed["batch_id"],
        "proof_length": len(proof),
        "computed_root": root,
        "root_on_ledger": bool(on_ledger.exists),
        "registered_by": on_ledger.org,
        "registered_at": on_ledger.block_ts,
        "verified": bool(genuine),
    })
    if genuine:
        result["reason"] = "document matches the registered evidence"
    elif not result["hash_matches_index"]:
        result["reason"] = "document content differs from what was registered"
    else:
        result["reason"] = "the off-chain index does not match the ledger (index altered, or evidence never anchored)"
    return result
