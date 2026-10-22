"""Turn a client's table into model-ready numeric arrays.

Each bank fits its own preprocessing (one-hot categories, standardization) on its OWN
training split only — exactly what a real bank would do. Nothing here is shared.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .partition import ID_COLUMNS
from .utils import load_json, load_table

# Long-tailed columns (e.g. src_txn_count: median 8, max 93,379 in HI-Small) get log1p before
# standardization, otherwise a few extreme accounts squeeze everyone else close to 0. This is a
# fixed public formula, so every bank applies it locally without sharing anything. On HI-Small
# it raised the centralized MLP's val PR-AUC from 0.12 to 0.26.
LOG1P_COLUMNS = ["src_txn_count", "src_fan_out", "src_burst", "src_secs_since_prev",
                 "dst_fan_in", "dst_burst", "account_age_days", "declared_annual_income",
                 "num_linked_accounts"]


@dataclass
class TabularPreprocessor:
    categorical: list[str]
    numeric: list[str]
    max_categories: int = 30
    # Category lists agreed in advance (e.g. features.SHARED_VOCAB). Columns listed here get
    # the same one-hot layout at every bank instead of one learned from the bank's own data.
    fixed_categories: dict[str, list[str]] = field(default_factory=dict)
    categories_: dict[str, list[str]] = field(default_factory=dict)
    mean_: pd.Series | None = None
    std_: pd.Series | None = None

    @classmethod
    def for_frame(cls, df: pd.DataFrame,
                  fixed_categories: dict[str, list[str]] | None = None) -> "TabularPreprocessor":
        cols = [c for c in df.columns if c not in ID_COLUMNS]
        cat = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        num = [c for c in cols if c not in cat]
        return cls(categorical=cat, numeric=num, fixed_categories=dict(fixed_categories or {}))

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        for c in self.categorical:
            if c in self.fixed_categories:
                self.categories_[c] = list(self.fixed_categories[c])
                continue
            top = df[c].astype(str).value_counts().index[: self.max_categories]
            self.categories_[c] = sorted(top)
        x = self._numeric(df)
        self.mean_ = x.mean()
        self.std_ = x.std().replace(0, 1.0).fillna(1.0)
        return self

    @property
    def feature_names(self) -> list[str]:
        names = list(self.numeric)
        for c in self.categorical:
            names += [f"{c}={v}" for v in self.categories_[c]] + [f"{c}=<other>"]
        return names

    def _numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[self.numeric].astype(np.float64)
        logged = [c for c in self.numeric if c in LOG1P_COLUMNS]
        x[logged] = np.log1p(x[logged].clip(lower=0))
        return x

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = [((self._numeric(df) - self.mean_) / self.std_).fillna(0).to_numpy()]
        for c in self.categorical:
            vals = df[c].astype(str)
            cats = self.categories_[c]
            onehot = np.zeros((len(df), len(cats) + 1), dtype=np.float64)
            idx = pd.Categorical(vals, categories=cats).codes  # -1 = unseen / rare
            onehot[np.arange(len(df)), np.where(idx >= 0, idx, len(cats))] = 1.0
            parts.append(onehot)
        return np.hstack(parts).astype(np.float32)


@dataclass
class ClientData:
    name: str
    X: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    txn_ids: dict[str, np.ndarray]
    feature_names: list[str]
    preprocessor: TabularPreprocessor

    @property
    def input_dim(self) -> int:
        return len(self.feature_names)


def load_client(client_dir: str | Path, core_only: bool = False,
                fixed_categories: dict[str, list[str]] | None = None) -> ClientData:
    """Load data/partitions/<client>/ and return train/val/test arrays.

    core_only: keep only the core columns every client has (schema.json: core_columns).
    fixed_categories: shared category lists, so one-hot columns line up across clients.
    """
    client_dir = Path(client_dir)
    df = load_table(client_dir / "transactions")
    meta = load_json(client_dir / "schema.json")
    cols = meta["core_columns"] if core_only else meta["columns"]
    df = df[ID_COLUMNS + cols]
    train = df[df["split"] == "train"]
    pre = TabularPreprocessor.for_frame(df, fixed_categories).fit(train)
    X, y, ids = {}, {}, {}
    for s in ("train", "val", "test"):
        part = df[df["split"] == s]
        X[s] = pre.transform(part)
        y[s] = part["label"].to_numpy(dtype=np.float32)
        ids[s] = part["txn_id"].to_numpy()
    return ClientData(client_dir.name, X, y, ids, pre.feature_names, pre)


def undersample(X: np.ndarray, y: np.ndarray, neg_per_pos: int = 20,
                seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Keep all positives and at most `neg_per_pos` negatives per positive (training only!)."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    k = min(len(neg), max(1, len(pos)) * neg_per_pos)
    keep = np.concatenate([pos, rng.choice(neg, size=k, replace=False)])
    rng.shuffle(keep)
    return X[keep], y[keep]
