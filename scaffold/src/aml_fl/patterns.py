"""Read the IBM '*_Patterns.txt' file and tag transactions with their laundering pattern.

The file lists laundering attempts, each as a block of transactions in the same CSV
format as '*_Trans.csv':

    BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out
    2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,2848.96,Euro,ACH,1
    ...
    END LAUNDERING ATTEMPT - FAN-OUT

Pattern types: FAN-OUT, FAN-IN, GATHER-SCATTER, SCATTER-GATHER, CYCLE, BIPARTITE, STACK,
RANDOM. Not every laundering transaction belongs to a listed attempt; those are tagged
"unlisted".

The tags are for EVALUATION ONLY (recall per pattern). They are derived from the labels,
so they must never be used as model features.
"""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

NONE = "none"          # not laundering
UNLISTED = "unlisted"  # laundering, but not part of any listed attempt

_BEGIN = re.compile(r"BEGIN LAUNDERING ATTEMPT\s*-\s*([A-Za-z\- ]+?)\s*(?::|$)")
_FIELDS = ["timestamp", "src_bank", "src_account", "dst_bank", "dst_account", "amount_received",
           "receiving_currency", "amount_paid", "payment_currency", "payment_format", "label"]


def default_path(trans_csv: str | Path) -> Path:
    """'data/raw/HI-Small_Trans.csv' -> 'data/raw/HI-Small_Patterns.txt'."""
    p = Path(trans_csv)
    return p.with_name(p.name.replace("_Trans.csv", "_Patterns.txt"))


def normalize_pattern(name: str) -> str:
    """'FAN-OUT' -> 'fan_out', 'Scatter-Gather' -> 'scatter_gather'."""
    return re.sub(r"[\s\-]+", "_", name.strip()).lower()


def parse_patterns(path: str | Path) -> pd.DataFrame:
    """Return one row per listed transaction with columns attempt_id, pattern, + _FIELDS."""
    rows: list[list] = []
    attempt, pattern = -1, None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("BEGIN"):
                m = _BEGIN.match(line)
                if not m:
                    raise ValueError(f"Cannot read pattern header: {line!r}")
                attempt += 1
                pattern = normalize_pattern(m.group(1))
            elif line.startswith("END"):
                pattern = None
            elif pattern is not None:
                fields = next(csv.reader([line]))
                if len(fields) != len(_FIELDS):
                    raise ValueError(f"Expected {len(_FIELDS)} fields, got {len(fields)}: {line!r}")
                rows.append([attempt, pattern, *fields])
    df = pd.DataFrame(rows, columns=["attempt_id", "pattern", *_FIELDS])
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M")
    # Bank IDs may be zero-padded in the file ("012"); data.clean stores them as ints.
    for side in ("src", "dst"):
        df[f"{side}_key"] = df[f"{side}_bank"].astype(int).astype(str) + "_" + df[f"{side}_account"].str.strip()
    df["amount_paid"] = df["amount_paid"].astype(float)
    log.info("Parsed %d attempts, %d transactions: %s", attempt + 1, len(df),
             df.groupby("pattern")["attempt_id"].nunique().to_dict())
    return df


_MATCH_ON = ["timestamp", "src_key", "dst_key", "amount_paid"]


def tag_transactions(df: pd.DataFrame, patterns: pd.DataFrame) -> pd.Series:
    """Pattern name for every row of a cleaned frame (see data.clean), aligned with df.

    Non-laundering rows get "none", laundering rows not found in the file get "unlisted".
    """
    keys = patterns[_MATCH_ON].copy()
    keys["amount_paid"] = keys["amount_paid"].round(2)
    keys["pattern"] = patterns["pattern"]
    keys = keys.drop_duplicates(_MATCH_ON)  # a transaction listed twice keeps its first pattern
    left = df[_MATCH_ON].copy()
    left["amount_paid"] = left["amount_paid"].round(2)
    tagged = left.merge(keys, on=_MATCH_ON, how="left")["pattern"].to_numpy()

    label = df["label"].to_numpy() == 1
    out = np.where(label, np.where(pd.isna(tagged), UNLISTED, tagged), NONE)
    matched = int((~pd.isna(tagged)).sum())
    if matched < len(keys):
        log.warning("%d of %d pattern transactions were not found in the transaction table "
                    "(expected only when using --nrows)", len(keys) - matched, len(keys))
    if (~pd.isna(tagged) & ~label).any():
        log.warning("%d pattern transactions are labelled non-laundering", int((~pd.isna(tagged) & ~label).sum()))
    return pd.Series(out, index=df.index, name="pattern")
