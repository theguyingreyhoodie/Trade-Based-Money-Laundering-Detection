"""Time-aware train / validation / test split.

Trades are split by invoice date: the oldest part of the period is training
data, the next part validation, the newest part test. The cut-off dates are
derived from the (public) data period only, so every FL client can apply the
same split without anyone pooling data.

"Stratified" is enforced as a guard: each split must contain enough cases of
every typology, otherwise :func:`check_stratification` raises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tbml.config import Config
from tbml.data.typologies import TYPOLOGY_CLASSES

SPLITS = ["train", "val", "test"]


def split_cutoffs(cfg: Config) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Last invoice date of the training period and of the validation period."""
    start = pd.Timestamp(cfg.data.start_date)
    end = pd.Timestamp(cfg.data.end_date)
    days = (end - start).days
    train_end = start + pd.Timedelta(days=int(round(days * cfg.split.train_frac)))
    val_end = start + pd.Timedelta(days=int(round(days * (cfg.split.train_frac + cfg.split.val_frac))))
    return train_end, val_end


def assign_split(dates: pd.Series, cfg: Config) -> np.ndarray:
    """Label each invoice date as ``train``, ``val`` or ``test``."""
    train_end, val_end = split_cutoffs(cfg)
    d = pd.to_datetime(dates).to_numpy()
    out = np.full(len(d), "test", dtype=object)
    out[d <= np.datetime64(val_end)] = "val"
    out[d <= np.datetime64(train_end)] = "train"
    return out


def split_summary(split: np.ndarray, y: np.ndarray, typology: np.ndarray) -> dict[str, dict]:
    """Size, positive rate and per-typology counts of each split."""
    summary: dict[str, dict] = {}
    for name in SPLITS:
        m = split == name
        counts = np.bincount(typology[m].astype(int), minlength=len(TYPOLOGY_CLASSES))
        summary[name] = {
            "n": int(m.sum()),
            "n_illicit": int(y[m].sum()),
            "illicit_rate": float(y[m].mean()) if m.any() else 0.0,
            "typology_counts": {TYPOLOGY_CLASSES[i]: int(c) for i, c in enumerate(counts) if i > 0},
        }
    return summary


def check_stratification(split: np.ndarray, y: np.ndarray, typology: np.ndarray, min_count: int) -> dict[str, dict]:
    """Raise if a split lacks a typology; return the summary otherwise."""
    summary = split_summary(split, y, typology)
    for name, info in summary.items():
        short = {t: c for t, c in info["typology_counts"].items() if c < min_count}
        if short:
            raise ValueError(f"split '{name}' has too few cases of {short} (minimum {min_count}); the split is not stratified enough")
    return summary
