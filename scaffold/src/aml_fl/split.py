"""Chronological train / validation / test split (no random shuffling -> no future leakage)."""
from __future__ import annotations

import pandas as pd


def time_split(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2) -> pd.Series:
    """Return a Series of 'train' / 'val' / 'test' aligned with df (which must be time-sorted).

    Cut points are placed at timestamp boundaries so that transactions sharing the same
    timestamp always land in the same split.
    """
    if not 0 < train_frac < 1 or not 0 <= val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("Need 0 < train_frac, 0 <= val_frac and train_frac + val_frac < 1")
    ts = df["timestamp"]
    t_train = ts.iloc[int(len(df) * train_frac)]
    t_val = ts.iloc[int(len(df) * (train_frac + val_frac))]
    split = pd.Series("test", index=df.index)
    split[ts < t_val] = "val"
    split[ts < t_train] = "train"
    return split
