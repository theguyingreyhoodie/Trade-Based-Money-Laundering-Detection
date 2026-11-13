"""Splitting trades among FL clients: IID, label/typology skew, quantity skew.

A client is an institution (or consortium) that holds the complete linked
records of its own customers' trades. The partitioner decides which trades
belong to which client; it is the simulation's stand-in for "who banks where".

Schemes
-------
``iid``        every client gets a random, equally sized share.
``dirichlet``  label and typology skew (Hsu et al., 2019): for every class
               (legitimate plus each typology) the share going to each client is
               drawn from Dirichlet(alpha). Small alpha means a client sees
               only a few typologies and a very different positive rate; it
               also makes client sizes unequal.
``quantity``   quantity skew only: client sizes follow Dirichlet(alpha), the
               class mix is the same everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tbml.data.typologies import TYPOLOGY_CLASSES

SCHEMES = ("iid", "dirichlet", "quantity")


@dataclass(frozen=True)
class Scenario:
    """One federation set-up: how many clients and how the data is split."""

    n_clients: int
    partition: str = "iid"
    alpha: float | None = None

    def __post_init__(self) -> None:
        if self.partition not in SCHEMES:
            raise ValueError(f"unknown partition scheme '{self.partition}' (choose from {SCHEMES})")
        if self.partition != "iid" and (self.alpha is None or self.alpha <= 0):
            raise ValueError(f"scheme '{self.partition}' needs alpha > 0")
        if self.n_clients < 2:
            raise ValueError("a federation needs at least 2 clients")

    @property
    def name(self) -> str:
        if self.partition == "iid":
            return f"n{self.n_clients}_iid"
        tag = "dir" if self.partition == "dirichlet" else "qty"
        return f"n{self.n_clients}_{tag}{self.alpha:g}"

    @property
    def label(self) -> str:
        if self.partition == "iid":
            return f"IID, {self.n_clients} clients"
        kind = "non-IID Dirichlet" if self.partition == "dirichlet" else "quantity skew"
        return f"{kind} α={self.alpha:g}, {self.n_clients} clients"

    @property
    def is_iid(self) -> bool:
        return self.partition == "iid"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "n_clients": self.n_clients, "partition": self.partition, "alpha": self.alpha, "label": self.label}


def _split_by_proportions(idx: np.ndarray, proportions: np.ndarray) -> list[np.ndarray]:
    cuts = (np.cumsum(proportions)[:-1] * len(idx)).astype(int)
    return np.split(idx, cuts)


def partition_trades(typology: np.ndarray, scenario: Scenario, seed: int, min_share: float = 0.25, max_tries: int = 20000) -> np.ndarray:
    """Client index (0..n_clients-1) for every trade.

    ``typology`` holds class indices (0 = legitimate).

    Sanity bound: with a 96% majority class, an unconstrained Dirichlet draw can
    leave a client with a few hundred trades that are almost all illicit, which
    is not a plausible institution. Draws in which a client would receive less
    than ``min_share`` of an equal share of the legitimate trades (or, for
    quantity skew, of all trades) are rejected. The typology classes are drawn
    without any constraint, so the typology mix is skewed exactly as alpha says.
    If no draw satisfies the bound, the most balanced one is used.
    """
    typology = np.asarray(typology, dtype=np.int64)
    n = len(typology)
    k = scenario.n_clients
    rng = np.random.default_rng(seed)

    if scenario.partition == "iid":
        owner = np.empty(n, dtype=np.int64)
        owner[rng.permutation(n)] = np.arange(n) % k
        return owner

    # size-driving proportions: rejection-sample until every client gets a workable share
    floor = min_share / k
    draws = rng.dirichlet(np.full(k, scenario.alpha), size=max_tries)
    ok = np.flatnonzero(draws.min(axis=1) >= floor)
    main = draws[ok[0]] if len(ok) else draws[int(draws.min(axis=1).argmax())]

    owner = np.empty(n, dtype=np.int64)
    if scenario.partition == "quantity":
        for client, part in enumerate(_split_by_proportions(rng.permutation(n), main)):
            owner[part] = client
        return owner

    for c in range(len(TYPOLOGY_CLASSES)):
        idx = rng.permutation(np.flatnonzero(typology == c))
        if not len(idx):
            continue
        proportions = main if c == 0 else rng.dirichlet(np.full(k, scenario.alpha))
        for client, part in enumerate(_split_by_proportions(idx, proportions)):
            owner[part] = client
    return owner


def partition_summary(owner: np.ndarray, typology: np.ndarray, split: np.ndarray | None = None) -> dict[str, Any]:
    """Sizes, positive rates and typology counts per client."""
    k = int(owner.max()) + 1
    clients = []
    for c in range(k):
        m = owner == c
        counts = np.bincount(typology[m].astype(int), minlength=len(TYPOLOGY_CLASSES))
        info: dict[str, Any] = {
            "client": c,
            "n": int(m.sum()),
            "n_illicit": int(counts[1:].sum()),
            "illicit_rate": float(counts[1:].sum() / max(m.sum(), 1)),
            "typology_counts": {TYPOLOGY_CLASSES[i]: int(v) for i, v in enumerate(counts) if i > 0},
            "n_typologies_present": int((counts[1:] > 0).sum()),
        }
        if split is not None:
            info["by_split"] = {name: {"n": int((m & (split == name)).sum()), "n_illicit": int((typology[m & (split == name)] > 0).sum())}
                                for name in ("train", "val", "test")}
        clients.append(info)
    sizes = np.array([c["n"] for c in clients], dtype=float)
    rates = np.array([c["illicit_rate"] for c in clients])
    return {
        "n_clients": k,
        "clients": clients,
        "size_imbalance": float(sizes.max() / max(sizes.min(), 1.0)),
        "illicit_rate_range": [float(rates.min()), float(rates.max())],
    }
