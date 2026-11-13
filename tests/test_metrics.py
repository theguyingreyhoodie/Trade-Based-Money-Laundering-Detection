"""Histogram-based metrics: agreement with scikit-learn, additivity, threshold selection."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from tbml.eval.metrics import N_CLASSES, ScoreHist, bin_threshold, evaluate_hists, evaluate_scores, score_bins


def _toy(n: int = 6000, seed: int = 0, prevalence: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.random(n) < prevalence
    logit = rng.normal(-3.0, 1.5, n) + 3.5 * y
    scores = 1.0 / (1.0 + np.exp(-logit))
    typology = np.where(y, rng.integers(1, N_CLASSES, n), 0)
    return scores, typology


def test_binning_is_monotone_and_bounded() -> None:
    p = np.array([0.0, 1e-9, 0.01, 0.5, 0.99, 1 - 1e-9, 1.0])
    bins = score_bins(p, 800)
    assert (np.diff(bins) >= 0).all()
    assert bins[0] == 0 and bins[-1] == 799
    assert bin_threshold(0) == 0.0 and bin_threshold(800) == 1.0
    assert abs(bin_threshold(400) - 0.5) < 1e-12
    # a score is flagged at threshold k exactly when its bin is >= k
    for k in (100, 400, 650):
        assert ((p >= bin_threshold(k)) == (bins >= k)).all()


def test_matches_sklearn() -> None:
    scores, typology = _toy()
    y = typology > 0
    hist = ScoreHist.from_scores(scores, typology)
    assert hist.n == len(y) and hist.n_pos == y.sum() and hist.n_neg == (~y).sum()
    assert abs(hist.pr_auc() - average_precision_score(y, scores)) < 0.005
    assert abs(hist.roc_auc() - roc_auc_score(y, scores)) < 0.002
    for k in (300, 400, 500):
        pred = scores >= bin_threshold(k)
        m = hist.at(k)
        assert m["confusion"] == {"tp": int((pred & y).sum()), "fp": int((pred & ~y).sum()), "fn": int((~pred & y).sum()), "tn": int((~pred & ~y).sum())}
        assert abs(m["f1"] - f1_score(y, pred)) < 1e-12


def test_best_threshold_maximises_f1() -> None:
    scores, typology = _toy(seed=1)
    hist = ScoreHist.from_scores(scores, typology)
    best = hist.best_threshold("f1")
    brute = max(hist.at(k)["f1"] for k in range(hist.n_bins + 1))
    assert abs(hist.at(best)["f1"] - brute) < 1e-12
    assert hist.at(hist.best_threshold("f2"))["recall"] >= hist.at(best)["recall"]
    with pytest.raises(ValueError):
        hist.best_threshold("accuracy")


def test_histograms_add_up_like_pooled_data() -> None:
    """Summing client histograms gives exactly the histogram of the pooled scores."""
    scores, typology = _toy(seed=2)
    pooled = ScoreHist.from_scores(scores, typology)
    parts = np.array_split(np.arange(len(scores)), 5)
    total = ScoreHist.empty()
    for idx in parts:
        total = total + ScoreHist.from_scores(scores[idx], typology[idx])
    np.testing.assert_array_equal(total.counts, pooled.counts)
    assert total.pr_auc() == pooled.pr_auc()
    restored = ScoreHist.from_list(pooled.to_list())
    np.testing.assert_array_equal(restored.counts, pooled.counts)
    assert len(pooled.to_list()) == N_CLASSES * pooled.n_bins


def test_per_typology_recall() -> None:
    scores, typology = _toy(seed=3)
    hist = ScoreHist.from_scores(scores, typology)
    k = 400
    m = hist.at(k)
    pred = scores >= bin_threshold(k)
    from tbml.data.typologies import TYPOLOGY_CLASSES

    for c in range(1, N_CLASSES):
        expected = pred[typology == c].mean()
        assert abs(m["recall_by_typology"][TYPOLOGY_CLASSES[c]] - expected) < 1e-12
        assert m["support_by_typology"][TYPOLOGY_CLASSES[c]] == int((typology == c).sum())
    overall = sum(m["recall_by_typology"][t] * m["support_by_typology"][t] for t in m["recall_by_typology"]) / hist.n_pos
    assert abs(overall - m["recall"]) < 1e-12


def test_threshold_comes_from_validation_only() -> None:
    val_s, val_t = _toy(seed=4)
    test_s, test_t = _toy(seed=5)
    a = evaluate_scores(val_s, val_t, test_s, test_t)
    # a completely different test set must not move the threshold
    b = evaluate_scores(val_s, val_t, 1.0 - test_s, test_t)
    assert a["threshold_bin"] == b["threshold_bin"]
    assert a["threshold"] == b["threshold"]
    assert a["f1"] != b["f1"]
    assert a["oracle_f1"] >= a["f1"] - 1e-12
    for key in ["precision", "recall", "f1", "pr_auc", "roc_auc", "fpr", "confusion", "recall_by_typology", "val"]:
        assert key in a


def test_degenerate_inputs() -> None:
    scores = np.array([0.1, 0.2, 0.3])
    no_pos = ScoreHist.from_scores(scores, np.zeros(3, dtype=int))
    assert np.isnan(no_pos.pr_auc()) and np.isnan(no_pos.roc_auc())
    assert no_pos.at(no_pos.best_threshold())["confusion"]["fp"] == 0, "with nothing to find, flag nothing"
    empty = ScoreHist.empty()
    assert empty.n == 0 and empty.at(0)["f1"] == 0.0
    perfect = ScoreHist.from_scores(np.array([0.01, 0.02, 0.98, 0.99]), np.array([0, 0, 1, 2]))
    assert perfect.pr_auc() == 1.0 and perfect.roc_auc() == 1.0
    assert evaluate_hists(perfect, perfect)["f1"] == 1.0
