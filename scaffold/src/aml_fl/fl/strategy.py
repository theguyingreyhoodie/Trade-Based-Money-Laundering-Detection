"""FedAvg that records a hash of every aggregated model, for the blockchain audit log.

Only the SHA-256 hash of the aggregated weights is recorded, never the weights themselves.
The audit layer (Phase 4) writes these hashes on-chain; anyone holding a model can later
re-hash it and check that it matches the recorded round.
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


class AuditedFedAvg(FedAvg):
    """FedAvg that keeps [{"round", "hash"}] for every aggregated model in `audit_log`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit_log: list[dict] = []

    def aggregate_train(self, server_round: int,
                        replies: Iterable[Message]) -> tuple[ArrayRecord | None, MetricRecord | None]:
        arrays, metrics = super().aggregate_train(server_round, replies)
        if arrays is not None:
            digest = hash_parameters(arrays)
            self.audit_log.append({"round": server_round, "hash": digest})
            log.info("round %d global model sha256=%s", server_round, digest)
        return arrays, metrics
