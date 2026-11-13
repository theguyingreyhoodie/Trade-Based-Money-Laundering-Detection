"""Registering model hashes of a federated run on the evidence ledger.

``ModelAuditSink`` is handed to the FL server as its audit sink. For every
round it registers the SHA-256 of each client's update under that client's
organisation and the SHA-256 of the aggregated global model under the
aggregator. Later, anyone holding a model file can prove which round of which
run it came from, or detect that it was altered.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from tbml.ledger.base import EvidenceLedger, EvidenceRecord, Receipt
from tbml.ledger.hashing import hash_arrays

MODEL_UPDATE = "model_update"
GLOBAL_MODEL = "global_model"


def client_org(client: int) -> str:
    return f"fl_client_{int(client)}"


class ModelAuditSink:
    """Callable audit sink: ``sink(entry)`` registers the entry's hash and annotates it with the receipt."""

    def __init__(self, ledger: EvidenceLedger, run_id: str, aggregator: str = "fl_aggregator") -> None:
        self.ledger = ledger
        self.run_id = run_id
        self.aggregator = aggregator
        self.receipts: list[Receipt] = []
        if aggregator not in ledger.organisations():
            ledger.register_organisation(aggregator)

    def ref_id(self, entry: dict[str, Any]) -> str:
        base = f"{self.run_id}/round-{int(entry['round']):03d}"
        return f"{base}/client-{int(entry['client'])}" if entry["kind"] == "client_update" else base

    def __call__(self, entry: dict[str, Any]) -> None:
        if entry["kind"] == "client_update":
            org, doc_type = client_org(entry["client"]), MODEL_UPDATE
            if org not in self.ledger.organisations():
                self.ledger.register_organisation(org)
        else:
            org, doc_type = self.aggregator, GLOBAL_MODEL
        receipt = self.ledger.register(entry["sha256"], doc_type, self.ref_id(entry), org)
        self.receipts.append(receipt)
        entry.update({"ledger_status": receipt.status, "tx_id": receipt.tx_id, "block": receipt.block, "ledger_org": org,
                      "ledger_cost": receipt.cost, "ledger_latency_s": receipt.latency_s})

    def summary(self) -> dict[str, Any]:
        latency = np.array([r.latency_s for r in self.receipts]) if self.receipts else np.zeros(1)
        cost = [r.cost for r in self.receipts if r.status == "registered"]
        return {
            "backend": self.ledger.backend,
            "run_id": self.run_id,
            "registered": sum(r.status == "registered" for r in self.receipts),
            "duplicates": sum(r.status == "duplicate" for r in self.receipts),
            "client_updates": sum(r.doc_type == MODEL_UPDATE for r in self.receipts),
            "global_models": sum(r.doc_type == GLOBAL_MODEL for r in self.receipts),
            "latency_ms_p50": float(np.percentile(latency, 50) * 1000),
            "latency_ms_p95": float(np.percentile(latency, 95) * 1000),
            "cost_total": int(sum(cost)),
            "cost_mean": float(np.mean(cost)) if cost else 0.0,
            "cost_unit": self.ledger.stats()["cost_unit"],
        }


def verify_model(ledger: EvidenceLedger, names: Sequence[str], weights: Sequence[np.ndarray]) -> tuple[str, EvidenceRecord]:
    """Hash a model as held now and look the hash up on the ledger."""
    digest = hash_arrays(zip(names, weights))
    return digest, ledger.verify(digest)
