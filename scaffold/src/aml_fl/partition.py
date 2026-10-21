"""Split the transactions into client banks and give each one its own schema.

A transaction belongs to the client that holds its *sending* bank (From Bank). Each
client keeps the core transaction features (shared by all clients) plus the history
features and profile columns listed for it in the config.
"""
from __future__ import annotations

import logging

import pandas as pd

from .features import TXN_FEATURES
from .profiles import PROFILE_COLUMNS

log = logging.getLogger(__name__)

ID_COLUMNS = ["txn_id", "timestamp", "src_key", "dst_key", "src_bank", "dst_bank", "split", "label"]


def assign_banks(df: pd.DataFrame, num_clients: int, mode: str = "group") -> dict[int, int]:
    """Map each From-Bank ID to a client index.

    group: greedy balancing — banks sorted by size are added to the currently smallest
           client, so clients have similar volume but different banks (non-IID).
    top:   only the `num_clients` largest banks become clients; other banks are dropped.
    """
    sizes = df["src_bank"].value_counts()  # sorted, largest first
    if mode == "top":
        return {bank: i for i, bank in enumerate(sizes.index[:num_clients])}
    if mode != "group":
        raise ValueError(f"Unknown partition mode: {mode}")
    totals = [0] * num_clients
    mapping: dict[int, int] = {}
    for bank, size in sizes.items():
        i = totals.index(min(totals))
        mapping[bank] = i
        totals[i] += size
    return mapping


def validate_schemas(schemas: dict, num_clients: int, core_features: list[str] = ()) -> None:
    bad_core = set(core_features) - set(TXN_FEATURES)
    if bad_core:
        raise ValueError(f"core_features: unknown features {sorted(bad_core)}")
    for i in range(num_clients):
        name = f"client_{i}"
        if name not in schemas:
            raise ValueError(f"No schema configured for {name}")
        bad_f = set(schemas[name].get("features", [])) - set(TXN_FEATURES)
        bad_p = set(schemas[name].get("profile", [])) - set(PROFILE_COLUMNS)
        if bad_f or bad_p:
            raise ValueError(f"{name}: unknown features {sorted(bad_f)} / profile columns {sorted(bad_p)}")


def partition(df: pd.DataFrame, num_clients: int, mode: str, schemas: dict,
              use_profiles: bool = True,
              core_features: list[str] = ()) -> tuple[dict[str, pd.DataFrame], dict[int, int]]:
    """Return {client_name: frame with ID columns + core + that client's own columns}, bank map."""
    validate_schemas(schemas, num_clients, core_features)
    bank_map = assign_banks(df, num_clients, mode)
    client_of = df["src_bank"].map(bank_map)
    clients: dict[str, pd.DataFrame] = {}
    for i in range(num_clients):
        name = f"client_{i}"
        own = schemas[name].get("features", [])
        cols = list(core_features) + [c for c in own if c not in core_features]
        if use_profiles:
            cols += list(schemas[name].get("profile", []))
        part = df.loc[client_of == i, ID_COLUMNS + cols].reset_index(drop=True)
        clients[name] = part
        log.info("%s: %d txns, %d laundering, %d banks, %d columns", name, len(part),
                 int(part["label"].sum()), sum(v == i for v in bank_map.values()), len(cols))
    return clients, bank_map
