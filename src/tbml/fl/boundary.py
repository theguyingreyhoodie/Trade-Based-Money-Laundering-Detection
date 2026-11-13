"""Client-boundary guard: what is allowed to leave a client.

Federated learning keeps raw data local only if the messages a client sends
cannot carry it. This module makes that checkable. Every reply that reaches the
server is validated against a whitelist before it is used:

* a training reply holds exactly one set of arrays whose names, shapes and
  dtypes equal the model template (so its size is fixed by the model, not by
  the client's data), plus a few scalar metrics and the SHA-256 of the update;
* an evaluation reply holds no arrays, only scalar metrics and two score
  histograms of fixed length;
* a statistics reply holds no arrays, only per-feature sums of fixed length.

Anything else (an extra record, an unknown key, an over-long list, a payload
larger than the fixed budget) raises :class:`BoundaryViolation`. The fixed-size
aggregates are the only information beyond model parameters that the server
receives; they are listed in ``docs/fl_design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from flwr.app import RecordDict
from flwr.common import serde

TRAIN, EVALUATE, STATS = "train", "evaluate", "stats"


class BoundaryViolation(RuntimeError):
    """A client reply contained something other than parameters and whitelisted aggregates."""


def message_bytes(content: RecordDict) -> int:
    """Serialised (protobuf) size of a message payload in bytes."""
    return int(serde.recorddict_to_proto(content).ByteSize())


@dataclass
class BoundarySpec:
    """Whitelist derived from the model template and the evaluation settings."""

    arrays: list[tuple[str, tuple[int, ...], str]]  # (name, shape, dtype) of the model
    n_features: int
    hist_len: int
    array_dtype_override: str | None = None  # "uint64" when updates are masked for secure aggregation
    extra_array_names: tuple[str, ...] = ()  # e.g. the masked weight slot of secure aggregation
    received: list[dict[str, Any]] = field(default_factory=list)

    def metric_limits(self, kind: str) -> dict[str, int]:
        """Allowed metric keys and the maximum list length of each (1 = scalar)."""
        common = {"num-examples": 1, "partition-id": 1}
        if kind == TRAIN:
            return {**common, "train-loss": 1, "steps": 1, "dp-epsilon": 1, "dp-noise": 1}
        if kind == EVALUATE:
            return {**common, "val-loss": 1, "hist-val": self.hist_len, "hist-test": self.hist_len}
        if kind == STATS:
            return {**common, "feat-sum": self.n_features, "feat-sumsq": self.n_features}
        raise ValueError(f"unknown reply kind '{kind}'")

    def config_limits(self, kind: str) -> dict[str, int]:
        """Allowed config keys and their maximum string length."""
        return {"update-sha256": 64} if kind == TRAIN else {}

    def model_bytes(self) -> int:
        """Raw size of the model arrays in bytes (4 bytes per float32 value)."""
        import numpy as np

        return int(sum(int(np.prod(shape)) * np.dtype(dtype).itemsize for _, shape, dtype in self.arrays))

    def byte_budget(self, kind: str) -> int:
        """Upper bound on the serialised size of a legitimate reply."""
        if kind == TRAIN:
            width = 2 if self.array_dtype_override == "uint64" else 1
            return int(self.model_bytes() * width * 1.05) + 256 * (len(self.arrays) + len(self.extra_array_names)) + 2048
        if kind == EVALUATE:
            return 2 * self.hist_len * 10 + 2048  # a varint-encoded count takes at most 10 bytes
        return 2 * self.n_features * 9 + 2048


def check_reply(content: RecordDict, spec: BoundarySpec, kind: str) -> dict[str, Any]:
    """Validate one client reply; return a receipt describing what was received.

    Raises :class:`BoundaryViolation` if the reply carries anything that is not
    whitelisted for its kind.
    """
    arrays = dict(content.array_records)
    metrics = dict(content.metric_records)
    configs = dict(content.config_records)

    def fail(reason: str) -> None:
        raise BoundaryViolation(f"{kind} reply rejected: {reason}")

    # ---- arrays: only the model, and only in training replies
    if kind == TRAIN:
        if len(arrays) != 1:
            fail(f"expected exactly one array record, got {len(arrays)}")
        record = next(iter(arrays.values()))
        expected = {name: (shape, dtype) for name, shape, dtype in spec.arrays}
        names = [k for k in record.keys() if k not in spec.extra_array_names]
        if names != list(expected):
            fail(f"array names differ from the model template: {sorted(set(names) ^ set(expected))[:4]}")
        for name, array in record.items():
            if name in spec.extra_array_names:
                if tuple(array.shape) != (1,):
                    fail(f"array '{name}' must hold one value")
                continue
            shape, dtype = expected[name]
            dtype = spec.array_dtype_override or dtype
            if tuple(array.shape) != tuple(shape) or str(array.dtype) != dtype:
                fail(f"array '{name}' is {array.dtype}{tuple(array.shape)}, template says {dtype}{tuple(shape)}")
    elif arrays:
        fail(f"{len(arrays)} array record(s) in a reply that must not contain arrays")

    # ---- metrics: whitelisted keys, bounded length, numbers only
    if len(metrics) != 1:
        fail(f"expected exactly one metric record, got {len(metrics)}")
    limits = spec.metric_limits(kind)
    record = next(iter(metrics.values()))
    for key, value in record.items():
        if key not in limits:
            fail(f"metric '{key}' is not whitelisted")
        size = len(value) if isinstance(value, list) else 1
        if size > limits[key]:
            fail(f"metric '{key}' has {size} values, at most {limits[key]} allowed")
        if isinstance(value, list) and limits[key] == 1:
            fail(f"metric '{key}' must be a scalar")
    if "num-examples" not in record:
        fail("metric 'num-examples' is missing")

    # ---- config records: only the update hash
    climits = spec.config_limits(kind)
    if len(configs) > (1 if climits else 0):
        fail(f"{len(configs)} config record(s), at most {1 if climits else 0} allowed")
    for record_c in configs.values():
        for key, value in record_c.items():
            if key not in climits:
                fail(f"config key '{key}' is not whitelisted")
            if not isinstance(value, str) or len(value) > climits[key]:
                fail(f"config value '{key}' must be a string of at most {climits[key]} characters")

    # ---- overall size: fixed by the model and the histogram resolution, never by the data
    size = message_bytes(content)
    if size > spec.byte_budget(kind):
        fail(f"payload of {size} bytes exceeds the budget of {spec.byte_budget(kind)} bytes")

    receipt = {
        "kind": kind,
        "bytes": size,
        "n_arrays": len(next(iter(arrays.values()))) if arrays else 0,
        "metric_keys": sorted(next(iter(metrics.values())).keys()),
        "config_keys": sorted(k for r in configs.values() for k in r.keys()),
    }
    spec.received.append(receipt)
    return receipt
