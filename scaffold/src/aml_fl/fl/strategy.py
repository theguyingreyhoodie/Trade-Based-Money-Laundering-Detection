"""FedAvg that records a hash of every aggregated model, for the blockchain audit log.

Only SHA-256 hashes are recorded, never the weights themselves: the aggregated model of every
round, and the update each bank sent in it. Each bank declares the hash of its own update in its
reply (an "audit" ConfigRecord); the server re-hashes what it received and refuses the round if
the two differ. The audit layer (Phase 4) writes these hashes on-chain; anyone holding a model
can later re-hash it and check that it matches the recorded round.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable

import numpy as np
from flwr.app import ArrayRecord, Message, MetricRecord
from flwr.serverapp.strategy import FedAvg

log = logging.getLogger(__name__)


def hash_parameters(arrays: ArrayRecord) -> str:
    """SHA-256 over every array's name, dtype, shape and bytes, in key order."""
    h = hashlib.sha256()
    for key, arr in zip(arrays.keys(), arrays.to_numpy_ndarrays()):
        arr = np.ascontiguousarray(arr)
        h.update(key.encode())
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


AUDIT_KEY = "audit"   # reply record: {"bank": name, "update-hash": sha256 of the sent arrays}


class AuditedFedAvg(FedAvg):
    """FedAvg that keeps [{"round", "hash", "updates": {bank: hash}}] in `audit_log`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit_log: list[dict] = []

    def aggregate_train(self, server_round: int,
                        replies: Iterable[Message]) -> tuple[ArrayRecord | None, MetricRecord | None]:
        replies = list(replies)
        updates = {}
        for msg in replies:
            if msg.has_error() or AUDIT_KEY not in msg.content:
                continue
            bank, declared = str(msg.content[AUDIT_KEY]["bank"]), msg.content[AUDIT_KEY]["update-hash"]
            received = hash_parameters(msg.content["arrays"])
            if received != declared:
                raise ValueError(f"Round {server_round}: the update received from {bank} does not match "
                                 f"the hash it declared ({received} != {declared})")
            updates[bank] = received
        arrays, metrics = super().aggregate_train(server_round, replies)
        if arrays is not None:
            digest = hash_parameters(arrays)
            self.audit_log.append({"round": server_round, "hash": digest, "updates": dict(sorted(updates.items()))})
            log.info("round %d global model sha256=%s", server_round, digest)
        return arrays, metrics
