"""Optional extensions of the federated protocol (stretch goals).

* **Secure aggregation** by pairwise additive masking (after Bonawitz et al.,
  2017, without dropout recovery): the server only ever sees masked updates and
  learns nothing but their sum.
* **Poisoning demo**: a malicious client boosts or flips its update, to show
  the effect of the norm-clipping defence in ``fl.server``.
* **DP-SGD** local training with Opacus (see :func:`dp_local_update`).

These are kept out of the core client/server code so the default path stays
small and easy to audit.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any

import numpy as np
from flwr.app import Array, ArrayRecord

from tbml.config import Config

#: fixed-point scale of the secure-aggregation encoding (values are multiplied by 2**FRACTION_BITS)
FRACTION_BITS = 24
N_SLOT = "__n__"


# --------------------------------------------------------------------------- secure aggregation


def _pair_mask(secret: str, server_round: int, a: int, b: int, size: int) -> np.ndarray:
    """Mask shared by clients ``a`` and ``b`` for one round: uniform over 64-bit integers."""
    low, high = sorted((int(a), int(b)))
    seed = hashlib.sha256(f"{secret}|{server_round}|{low}|{high}".encode()).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(seed[:16], "big")))
    return rng.integers(0, np.iinfo(np.uint64).max, size=size, dtype=np.uint64, endpoint=True)


def _encode(values: np.ndarray) -> np.ndarray:
    """Fixed-point encode real values into the ring of 64-bit integers."""
    return np.rint(np.asarray(values, dtype=np.float64) * (1 << FRACTION_BITS)).astype(np.int64).view(np.uint64)


def _decode(ring: np.ndarray) -> np.ndarray:
    return ring.view(np.int64).astype(np.float64) / (1 << FRACTION_BITS)


def mask_vector(vector: np.ndarray, client: int, participants: list[int], server_round: int, secret: str) -> np.ndarray:
    """Encode ``vector`` and add this client's pairwise masks (mod 2**64).

    For every other participant ``j`` the two clients derive the same mask from
    their shared secret; the one with the smaller id adds it, the other
    subtracts it. Summed over all participants the masks cancel exactly.
    """
    out = _encode(vector)
    with np.errstate(over="ignore"):
        for other in participants:
            if other == client:
                continue
            mask = _pair_mask(secret, server_round, client, other, out.size)
            out = out + mask if client < other else out - mask
    return out


def mask_update(names: list[str], weights: list[np.ndarray], metrics: dict[str, Any], client: int, server_round: int,
                config: Any, options: dict[str, Any]) -> ArrayRecord:
    """Client side: send ``n_i * w_i`` (and ``n_i`` itself) under pairwise masks.

    ``options['group_secret']`` stands in for the pairwise keys that real
    clients would agree on with a Diffie-Hellman exchange. It exists only in
    client code; the server never reads it.
    """
    participants = [int(p) for p in config["secagg-participants"]]
    n = float(metrics["num-examples"])
    flat = np.concatenate([np.asarray(w, dtype=np.float64).ravel() * n for w in weights] + [np.array([n])])
    masked = mask_vector(flat, client, participants, server_round, str(options["group_secret"]))
    record: "OrderedDict[str, Array]" = OrderedDict()
    offset = 0
    for name, w in zip(names, weights):
        size = int(np.asarray(w).size)
        record[name] = Array(masked[offset : offset + size].reshape(np.asarray(w).shape))
        offset += size
    record[N_SLOT] = Array(masked[offset : offset + 1])
    return ArrayRecord(array_dict=record)


def unmask_sum(records: list[ArrayRecord], names: list[str], options: dict[str, Any]) -> list[np.ndarray]:
    """Server side: add the masked updates; the masks cancel and the weighted mean remains."""
    with np.errstate(over="ignore"):
        totals = {name: sum((r[name].numpy().astype(np.uint64) for r in records[1:]), records[0][name].numpy().astype(np.uint64))
                  for name in [*names, N_SLOT]}
    n_total = float(_decode(totals[N_SLOT])[0])
    if n_total <= 0:
        raise RuntimeError("secure aggregation produced a non-positive example count; a participant is missing")
    return [(_decode(totals[name]) / n_total).astype(np.float32) for name in names]


# --------------------------------------------------------------------------- poisoning demo


def poison_update(global_weights: list[np.ndarray], local_weights: list[np.ndarray], options: dict[str, Any]) -> list[np.ndarray]:
    """A malicious client's update: boosted (``scale``) and optionally sign-flipped.

    The first two arrays are the shared feature scaler and are left alone.
    """
    scale = float(options.get("scale", 10.0)) * (-1.0 if options.get("mode", "boost") == "signflip" else 1.0)
    out = []
    for i, (g, l) in enumerate(zip(global_weights, local_weights)):
        out.append(l if i < 2 else (g + scale * (l - g)).astype(np.float32))
    return out


# --------------------------------------------------------------------------- DP-SGD


def dp_local_update(cfg: Config, fs: Any, global_weights: list[np.ndarray], partition_id: int, server_round: int, seed: int,
                    config: Any, options: dict[str, Any]) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Local training with DP-SGD (per-example clipping + Gaussian noise) using Opacus."""
    from tbml.fl.dp import dp_train_round

    return dp_train_round(cfg, fs, global_weights, partition_id, server_round, seed, lr=float(config["lr"]),
                          local_epochs=int(config["local-epochs"]), noise_multiplier=float(options["noise_multiplier"]),
                          max_grad_norm=float(options["max_grad_norm"]))
