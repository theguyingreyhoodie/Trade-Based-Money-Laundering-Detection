"""Phase 2 starter: quick baselines to check that the prepared data carries signal.

    1. Centralized  - all data pooled, every feature (upper bound)
    2. Local-only   - each client trains on its own data and schema only (lower bound)

Models: Logistic Regression, XGBoost (if installed) and the PyTorch MLP (if installed).
The MLP is the same BankModel (encoder + head) that federated learning trains, so its
centralized / local-only numbers are the bounds the FL result should land between.

Results go to experiments/results/baselines.csv. If data/processed/patterns.parquet exists
(see prepare_data.py), recall per laundering pattern goes to baselines_by_pattern.csv.

Usage:
    python scripts/run_baselines.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

try:  # Load xgboost BEFORE torch: on macOS their two OpenMP runtimes segfault the other way round.
    import xgboost  # noqa: F401
except Exception:
    pass
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aml_fl.metrics import best_f1_threshold, evaluate, recall_by_pattern  # noqa: E402
from aml_fl.preprocess import TabularPreprocessor, load_client, undersample  # noqa: E402
from aml_fl.utils import load_config, load_table, set_seed, setup_logging  # noqa: E402

log = logging.getLogger("baselines")


class TorchMLP:
    """sklearn-style wrapper around models.BankModel (fit / predict_proba).

    Trains with early stopping on the validation split (see models.train_early_stopping).
    """

    def __init__(self, max_epochs: int = 60, patience: int = 8, lr: float = 1e-3, batch_size: int = 512):
        self.max_epochs, self.patience, self.lr, self.batch_size = max_epochs, patience, lr, batch_size

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> "TorchMLP":
        import torch

        from aml_fl.models import BankModel, pos_weight, train_early_stopping

        # xgboost and torch each bring their own OpenMP runtime; once xgboost has run, torch's
        # multi-threaded ops deadlock on macOS. One thread avoids OpenMP and is plenty for this MLP.
        torch.set_num_threads(1)
        y = y.astype(np.float32)
        self.model = BankModel(input_dim=X.shape[1])
        self.best_val_pr_auc_, self.best_epoch_ = train_early_stopping(
            self.model, X.astype(np.float32), y, X_val.astype(np.float32), y_val.astype(np.float32),
            max_epochs=self.max_epochs, patience=self.patience, lr=self.lr, batch_size=self.batch_size,
            weight=pos_weight(y))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        from aml_fl.models import predict_proba

        p = predict_proba(self.model, X.astype(np.float32))
        return np.column_stack([1 - p, p])


def models(seed: int, mlp_cfg: dict | None = None) -> dict:
    out = {"logreg": lambda: LogisticRegression(max_iter=2000, class_weight="balanced")}
    try:
        from xgboost import XGBClassifier

        out["xgboost"] = lambda: XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                               subsample=0.8, colsample_bytree=0.8,
                                               eval_metric="aucpr", random_state=seed, n_jobs=-1)
    except Exception as e:  # ImportError, or XGBoostError when libomp is missing on macOS
        log.warning("xgboost unavailable (%s); skipping it. On macOS: brew install libomp",
                    type(e).__name__)
    try:
        import torch  # noqa: F401

        out["mlp"] = lambda: TorchMLP(**(mlp_cfg or {}))
    except ImportError:
        log.warning("torch not installed; skipping the MLP")
    return out


def fit_eval(make, X, y, seed) -> tuple[dict, np.ndarray | None, float]:
    """Train, tune the threshold on val, evaluate on test. Returns metrics, test scores, threshold."""
    Xtr, ytr = undersample(X["train"], y["train"], neg_per_pos=20, seed=seed)
    if ytr.sum() == 0:
        return {"note": "no positives in train"}, None, 0.5
    set_seed(seed)
    clf = make()
    if isinstance(clf, TorchMLP):
        clf.fit(Xtr, ytr, X["val"], y["val"])
    else:
        clf.fit(Xtr, ytr)
    thr = best_f1_threshold(y["val"], clf.predict_proba(X["val"])[:, 1]) if y["val"].sum() else 0.5
    scores = clf.predict_proba(X["test"])[:, 1]
    return evaluate(y["test"], scores, thr), scores, thr


def pattern_rows(pattern_of: pd.Series | None, txn_ids, scores, thr, **info) -> list[dict]:
    if pattern_of is None or scores is None:
        return []
    per = recall_by_pattern(pattern_of.reindex(txn_ids).fillna("none").to_numpy(), scores, thr)
    return [{**info, **r} for r in per.to_dict("records")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(ROOT / args.config)
    set_seed(cfg["seed"])
    rows, by_pattern = [], []
    candidates = models(cfg["seed"], cfg.get("mlp"))

    proc = ROOT / cfg["data"]["processed_dir"]
    try:
        pattern_of = load_table(proc / "patterns").set_index("txn_id")["pattern"]
    except FileNotFoundError:
        pattern_of = None
        log.info("No patterns table; skipping per-pattern recall")

    # 1. Centralized
    df = load_table(proc / "all_transactions")
    pre = TabularPreprocessor.for_frame(df).fit(df[df["split"] == "train"])
    X = {s: pre.transform(df[df["split"] == s]) for s in ("train", "val", "test")}
    y = {s: df.loc[df["split"] == s, "label"].to_numpy() for s in ("train", "val", "test")}
    test_ids = df.loc[df["split"] == "test", "txn_id"].to_numpy()
    for name, make in candidates.items():
        res, scores, thr = fit_eval(make, X, y, cfg["seed"])
        rows.append({"setup": "centralized", "client": "all", "model": name, **res})
        by_pattern += pattern_rows(pattern_of, test_ids, scores, thr, setup="centralized", client="all", model=name)
        log.info("centralized %-8s %s", name, res)

    # 2. Local-only
    for cdir in sorted((ROOT / cfg["data"]["partitions_dir"]).glob("client_*")):
        c = load_client(cdir)
        for name, make in candidates.items():
            res, scores, thr = fit_eval(make, c.X, c.y, cfg["seed"])
            rows.append({"setup": "local", "client": c.name, "model": name, **res})
            by_pattern += pattern_rows(pattern_of, c.txn_ids["test"], scores, thr,
                                       setup="local", client=c.name, model=name)
            log.info("local %s %-8s %s", c.name, name, res)

    out = ROOT / "experiments" / "results" / "baselines.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("Saved %s", out)
    if by_pattern:
        pd.DataFrame(by_pattern).to_csv(out.with_name("baselines_by_pattern.csv"), index=False)
        log.info("Saved %s", out.with_name("baselines_by_pattern.csv"))


if __name__ == "__main__":
    main()
