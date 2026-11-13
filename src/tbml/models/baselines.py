"""Baselines (Step 03, objective 2): rules engine and centralised ML on pooled data.

Centralised training is the upper bound that a federation tries to approach:
one party holds every record. Models: logistic regression, random forest,
LightGBM, and the MLP that is later federated.

Protocol for every method: fit on the training period, pick the decision
threshold on the validation period only, report on the test period.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from tbml.config import Config
from tbml.eval.metrics import evaluate_scores
from tbml.features.build import FeatureSet
from tbml.models.mlp import fit_mlp, predict_proba
from tbml.rules.engine import RULES, rule_hits, rule_score
from tbml.utils import derive_seed, get_logger, save_json

log = get_logger(__name__)

MODEL_NAMES = ["logreg", "random_forest", "lightgbm", "mlp"]
METHOD_LABELS = {
    "rules": "Rules engine",
    "logreg": "Logistic regression",
    "random_forest": "Random forest",
    "lightgbm": "LightGBM",
    "mlp": "MLP (centralised)",
}


def _fit_logreg(cfg: Config, train: FeatureSet, val: FeatureSet, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    p = cfg.models.logreg
    model = make_pipeline(StandardScaler(), LogisticRegression(C=float(p.C), max_iter=int(p.max_iter), class_weight="balanced", random_state=seed))
    model.fit(train.X, train.y)
    return model, lambda X: model.predict_proba(X)[:, 1], {}


def _fit_random_forest(cfg: Config, train: FeatureSet, val: FeatureSet, seed: int):
    from sklearn.ensemble import RandomForestClassifier

    p = cfg.models.random_forest
    model = RandomForestClassifier(
        n_estimators=int(p.n_estimators), max_depth=None if p.max_depth is None else int(p.max_depth),
        min_samples_leaf=int(p.min_samples_leaf), class_weight=p.get("class_weight", "balanced"), n_jobs=-1, random_state=seed,
    )
    model.fit(train.X, train.y)
    return model, lambda X: model.predict_proba(X)[:, 1], {}


def _fit_lightgbm(cfg: Config, train: FeatureSet, val: FeatureSet, seed: int):
    from tbml.models.gbm import LGBMClassifier

    p = cfg.models.lightgbm
    ratio = float((train.y == 0).sum() / max((train.y == 1).sum(), 1))
    model = LGBMClassifier(
        n_estimators=int(p.n_estimators), learning_rate=float(p.learning_rate), num_leaves=int(p.num_leaves),
        min_child_samples=int(p.min_child_samples), subsample=float(p.subsample), subsample_freq=1,
        colsample_bytree=float(p.colsample_bytree), scale_pos_weight=float(np.sqrt(ratio)), random_state=seed, verbose=-1,
    )
    model.fit(train.X, train.y)
    gain = model.booster_.feature_importance(importance_type="gain")
    top = sorted(zip(train.feature_names, gain / max(gain.sum(), 1e-12)), key=lambda kv: -kv[1])[:20]
    return model, lambda X: model.predict_proba(X)[:, 1], {"top_features_by_gain": [{"feature": k, "share": float(v)} for k, v in top]}


def _fit_mlp(cfg: Config, train: FeatureSet, val: FeatureSet, seed: int):
    model, info = fit_mlp(cfg, train.X, train.y, val.X, val.y, seed)
    return model, lambda X: predict_proba(model, X), {"best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"]}


FITTERS: dict[str, Callable] = {"logreg": _fit_logreg, "random_forest": _fit_random_forest, "lightgbm": _fit_lightgbm, "mlp": _fit_mlp}


def rules_baseline(cfg: Config, fs: FeatureSet) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate the rules engine; returns (metrics, hits, score) for all trades."""
    hits = rule_hits(fs.X, cfg, fs.feature_names)
    score = rule_score(hits, cfg)
    val, test = fs.split == "val", fs.split == "test"
    metric = str(cfg.models.threshold_metric)
    result = evaluate_scores(score[val], fs.typology[val], score[test], fs.typology[test], metric)
    # per-rule quality on the test period: how often it fires and how often it is right
    y_test = fs.y[test]
    per_rule = []
    for j, rule in enumerate(RULES):
        h = hits[test, j]
        own = fs.typology[test] == _typology_index(rule.typology)
        per_rule.append({
            "rule": rule.rule_id, "name": rule.name, "typology": rule.typology, "hits": int(h.sum()),
            "precision": float(y_test[h].mean()) if h.any() else None,
            "recall_own_typology": float(h[own].mean()) if own.any() else None,
        })
    result["per_rule"] = per_rule
    result["any_rule"] = {"alert_rate": float(hits[test].any(axis=1).mean()), "recall": float(hits[test][y_test == 1].any(axis=1).mean()),
                          "precision": float(y_test[hits[test].any(axis=1)].mean())}
    return result, hits, score


def _typology_index(name: str) -> int:
    from tbml.data.typologies import TYPOLOGY_CLASSES

    return TYPOLOGY_CLASSES.index(name)


def save_model(name: str, model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if name == "mlp":
        torch.save(model.state_dict(), path.with_suffix(".pt"))
    else:
        import joblib

        joblib.dump(model, path.with_suffix(".joblib"))


def run_baselines(cfg: Config, fs: FeatureSet, seed: int, out_dir: Path | None = None, models: list[str] | None = None) -> dict[str, Any]:
    """Train and evaluate the rules engine and all centralised models for one seed."""
    train, val, test = fs.part("train"), fs.part("val"), fs.part("test")
    metric = str(cfg.models.threshold_metric)
    results: dict[str, Any] = {}
    scores: dict[str, np.ndarray] = {}

    results["rules"], hits, rscore = rules_baseline(cfg, fs)
    scores["rules"] = rscore
    log.info("seed %d  %-14s F1 %.3f  P %.3f  R %.3f  PR-AUC %.3f", seed, "rules", results["rules"]["f1"], results["rules"]["precision"],
             results["rules"]["recall"], results["rules"]["pr_auc"])

    for name in models or MODEL_NAMES:
        import time

        start = time.perf_counter()
        model, predict, extra = FITTERS[name](cfg, train, val, derive_seed(seed, "baseline", name) % (2**31))
        fit_seconds = time.perf_counter() - start
        all_scores = predict(fs.X).astype(np.float64)
        mask_val, mask_test = fs.split == "val", fs.split == "test"
        result = evaluate_scores(all_scores[mask_val], val.typology, all_scores[mask_test], test.typology, metric)
        result.update(extra)
        result["fit_seconds"] = fit_seconds
        results[name] = result
        scores[name] = all_scores
        log.info("seed %d  %-14s F1 %.3f  P %.3f  R %.3f  PR-AUC %.3f  (%.1fs)", seed, name, result["f1"], result["precision"], result["recall"],
                 result["pr_auc"], fit_seconds)
        if out_dir is not None:
            save_model(name, model, out_dir / name)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "scores.npz", trade_id=fs.trade_id.astype("U16"), split=fs.split.astype("U5"), y=fs.y,
                            typology=fs.typology, rule_hits=hits, **{f"score_{k}": v for k, v in scores.items()})
    return results


def run_and_save(cfg: Config, fs: FeatureSet, seed: int) -> dict[str, Any]:
    """Run the baselines for one seed and write metrics to ``results/baselines``."""
    out_dir = cfg.seed_dir("artifacts_dir", seed, "baselines")
    results = run_baselines(cfg, fs, seed, out_dir)
    payload = {"seed": seed, "n_features": len(fs.feature_names), "methods": results}
    save_json(payload, cfg.path("results_dir", "baselines", f"seed_{seed}.json"))
    return payload


def load_baseline_scores(cfg: Config, seed: int) -> dict[str, np.ndarray]:
    """Scores, rule hits and labels saved by :func:`run_baselines` for one seed."""
    with np.load(cfg.seed_dir("artifacts_dir", seed, "baselines", "scores.npz"), allow_pickle=False) as z:
        return {key: z[key] for key in z.files}
