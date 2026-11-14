"""Feature pipeline entry points: build, store and describe the centralised feature set."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tbml.config import Config
from tbml.data.generator import TradeDataset, load_dataset
from tbml.data.typologies import TYPOLOGY_CLASSES
from tbml.features.build import FeatureSet, Reference, build_features
from tbml.features.dictionary import FEATURE_BY_NAME
from tbml.features.link import link_dataset
from tbml.models.split import check_stratification


def central_path(cfg: Config, seed: int) -> Path:
    return cfg.seed_dir("artifacts_dir", seed, "features", "central.npz")


def build_central_features(cfg: Config, seed: int, ds: TradeDataset | None = None) -> FeatureSet:
    """Features of the pooled data: what a single party holding everything would compute."""
    ds = ds or load_dataset(cfg.seed_dir("data_dir", seed))
    linked = link_dataset(ds)
    ref = Reference.from_tables(ds.tables)
    fs = build_features(linked, ds.tables["kyc_entities"], ds.tables["banks"], ref, cfg, ds.tables["labels"])
    fs.save(central_path(cfg, seed))
    return fs


def load_central_features(cfg: Config, seed: int) -> FeatureSet:
    path = central_path(cfg, seed)
    if not path.is_file():
        return build_central_features(cfg, seed)
    return FeatureSet.load(path)


def rank_auc(score: np.ndarray, positive: np.ndarray) -> float:
    """ROC-AUC of one score via the rank-sum statistic (ties get average ranks)."""
    from scipy.stats import rankdata

    n_pos = int(positive.sum())
    n_neg = int(len(positive) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def feature_report(fs: FeatureSet, cfg: Config) -> dict[str, Any]:
    """Descriptive check of the indicators: do they respond to the typology they target?

    For every typology, each feature's one-dimensional ROC-AUC against legitimate
    trades is computed on the training period. ``max(auc, 1 - auc)`` is reported
    as separation, since some indicators are low for illicit trades.
    """
    train = fs.part("train")
    legit = train.typology == 0
    report: dict[str, Any] = {
        "n_features": len(fs.feature_names),
        "n_trades": len(fs),
        "split": check_stratification(fs.split, fs.y, fs.typology, int(cfg.split.min_typology_count)),
        "features": {},
        "typologies": {},
    }
    sep = np.zeros((len(TYPOLOGY_CLASSES), len(fs.feature_names)))
    for k in range(1, len(TYPOLOGY_CLASSES)):
        rows = legit | (train.typology == k)
        positive = train.typology[rows] == k
        for j in range(len(fs.feature_names)):
            auc = rank_auc(train.X[rows, j], positive)
            sep[k, j] = max(auc, 1 - auc) if np.isfinite(auc) else 0.5
    for j, name in enumerate(fs.feature_names):
        spec = FEATURE_BY_NAME[name]
        auc_all = rank_auc(train.X[:, j], train.y == 1)
        report["features"][name] = {
            "group": spec.group,
            "mean_legit": float(train.X[legit, j].mean()),
            "mean_illicit": float(train.X[~legit, j].mean()),
            "auc_any_illicit": auc_all,
            "best_typology": TYPOLOGY_CLASSES[int(sep[1:, j].argmax()) + 1],
            "best_typology_separation": float(sep[1:, j].max()),
        }
    for k in range(1, len(TYPOLOGY_CLASSES)):
        top = np.argsort(-sep[k])[:6]
        targeted = [j for j, name in enumerate(fs.feature_names) if TYPOLOGY_CLASSES[k] in FEATURE_BY_NAME[name].typologies]
        best_targeted = sorted(targeted, key=lambda j: -sep[k, j])[:4]
        report["typologies"][TYPOLOGY_CLASSES[k]] = {
            "top_features": [{"feature": fs.feature_names[j], "separation": float(sep[k, j])} for j in top],
            "top_targeted": [{"feature": fs.feature_names[j], "separation": float(sep[k, j])} for j in best_targeted],
            "best_targeted_separation": float(sep[k, targeted].max()) if targeted else None,
        }
    report["separation_matrix"] = {TYPOLOGY_CLASSES[k]: [round(float(v), 4) for v in sep[k]] for k in range(1, len(TYPOLOGY_CLASSES))}
    return report
