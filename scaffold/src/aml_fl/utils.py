"""Small shared helpers: config loading, seeding, table I/O."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger("aml_fl")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config. `extends: other.yaml` (relative to this file) loads that one first and
    overrides it key by key, so an experiment config only lists what it changes."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    return _merge(load_config(path.parent / parent), cfg) if parent else cfg


def results_dir(cfg: dict, root: str | Path) -> Path:
    """Where scripts write results: data.results_dir (default experiments/results), under root."""
    return Path(root) / cfg["data"].get("results_dir", "experiments/results")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional for the data pipeline
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Save as Parquet (fast, typed). Falls back to CSV if pyarrow is not installed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _has_parquet():
        path = path.with_suffix(".parquet")
        df.to_parquet(path, index=False)
    else:
        path = path.with_suffix(".csv")
        log.warning("pyarrow not installed; writing CSV instead: %s", path)
        df.to_csv(path, index=False)
    return path


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a table saved by save_table (accepts a path with or without extension)."""
    path = Path(path)
    for p in (path, path.with_suffix(".parquet"), path.with_suffix(".csv")):
        if p.exists():
            return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    raise FileNotFoundError(path)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
