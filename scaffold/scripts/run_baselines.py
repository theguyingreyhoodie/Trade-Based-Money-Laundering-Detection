"""Phase 2 starter: quick baselines to check that the prepared data carries signal.

    1. Centralized  - all data pooled, every feature (upper bound)
    2. Local-only   - each client trains on its own data and schema only (lower bound)

Models: Logistic Regression and (if installed) XGBoost. Results go to
experiments/results/baselines.csv. The PyTorch MLP versions come next, in Phase 2/3.

Usage:
    python scripts/run_baselines.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aml_fl.metrics import best_f1_threshold, evaluate  # noqa: E402
from aml_fl.partition import ID_COLUMNS  # noqa: E402
from aml_fl.preprocess import TabularPreprocessor, load_client, undersample  # noqa: E402
from aml_fl.utils import load_config, load_table, set_seed, setup_logging  # noqa: E402

log = logging.getLogger("baselines")


def models(seed: int) -> dict:
    out = {"logreg": lambda: LogisticRegression(max_iter=2000, class_weight="balanced")}
    try:
        from xgboost import XGBClassifier

        out["xgboost"] = lambda: XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                               subsample=0.8, colsample_bytree=0.8,
                                               eval_metric="aucpr", random_state=seed, n_jobs=-1)
    except ImportError:
        log.warning("xgboost not installed; skipping it")
    return out


def fit_eval(make, X, y, seed) -> dict:
    Xtr, ytr = undersample(X["train"], y["train"], neg_per_pos=20, seed=seed)
    if ytr.sum() == 0:
        return {"note": "no positives in train"}
    clf = make().fit(Xtr, ytr)
    thr = best_f1_threshold(y["val"], clf.predict_proba(X["val"])[:, 1]) if y["val"].sum() else 0.5
    return evaluate(y["test"], clf.predict_proba(X["test"])[:, 1], thr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(ROOT / args.config)
    set_seed(cfg["seed"])
    rows = []

    # 1. Centralized
    df = load_table(ROOT / cfg["data"]["processed_dir"] / "all_transactions")
    pre = TabularPreprocessor.for_frame(df).fit(df[df["split"] == "train"])
    X = {s: pre.transform(df[df["split"] == s]) for s in ("train", "val", "test")}
    y = {s: df.loc[df["split"] == s, "label"].to_numpy() for s in ("train", "val", "test")}
    for name, make in models(cfg["seed"]).items():
        res = fit_eval(make, X, y, cfg["seed"])
        rows.append({"setup": "centralized", "client": "all", "model": name, **res})
        log.info("centralized %-8s %s", name, res)

    # 2. Local-only
    for cdir in sorted((ROOT / cfg["data"]["partitions_dir"]).glob("client_*")):
        c = load_client(cdir)
        for name, make in models(cfg["seed"]).items():
            res = fit_eval(make, c.X, c.y, cfg["seed"])
            rows.append({"setup": "local", "client": c.name, "model": name, **res})
            log.info("local %s %-8s %s", c.name, name, res)

    out = ROOT / "experiments" / "results" / "baselines.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("Saved %s", out)


if __name__ == "__main__":
    main()
