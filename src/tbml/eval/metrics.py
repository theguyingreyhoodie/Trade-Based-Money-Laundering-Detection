"""Detection metrics computed from score histograms.

Every method in the project (rules, centralised models, local-only models and
the federated model) is scored through the same code path: scores are binned
into a per-class histogram, and all metrics are derived from the histogram.

Why histograms: in a federation nobody may pool predictions with labels. A
histogram of scores per class is an aggregate that each client can compute
locally and the server can sum. Threshold selection and every metric below need
nothing else, so centralised and federated results are computed identically.

Bins are uniform on the logit scale, which keeps resolution where sigmoid
outputs concentrate (close to 0 and 1). A decision threshold is a bin edge, so
confusion counts at a threshold are exact; PR-AUC and ROC-AUC treat scores in
one bin as tied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tbml.data.typologies import TYPOLOGY_CLASSES

DEFAULT_BINS = 800
LOGIT_RANGE = 12.0
N_CLASSES = len(TYPOLOGY_CLASSES)


def score_bins(scores: np.ndarray, n_bins: int = DEFAULT_BINS) -> np.ndarray:
    """Bin index of each probability-like score in [0, 1]."""
    p = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    with np.errstate(divide="ignore"):
        logit = np.log(p) - np.log1p(-p)
    logit = np.clip(np.nan_to_num(logit, posinf=LOGIT_RANGE, neginf=-LOGIT_RANGE), -LOGIT_RANGE, LOGIT_RANGE)
    idx = np.floor((logit + LOGIT_RANGE) / (2 * LOGIT_RANGE) * n_bins).astype(np.int64)
    return np.clip(idx, 0, n_bins - 1)


def bin_threshold(k: int, n_bins: int = DEFAULT_BINS) -> float:
    """Probability at the lower edge of bin ``k``: scores in bins >= k are flagged."""
    if k <= 0:
        return 0.0
    if k >= n_bins:
        return 1.0
    return float(1.0 / (1.0 + np.exp(-(-LOGIT_RANGE + 2 * LOGIT_RANGE * k / n_bins))))


@dataclass
class ScoreHist:
    """Counts of scores per typology class (row 0 = legitimate) and score bin."""

    counts: np.ndarray  # int64 [N_CLASSES, n_bins]

    # ------------------------------------------------------------------ construction
    @classmethod
    def empty(cls, n_bins: int = DEFAULT_BINS) -> "ScoreHist":
        return cls(np.zeros((N_CLASSES, n_bins), dtype=np.int64))

    @classmethod
    def from_scores(cls, scores: np.ndarray, typology: np.ndarray, n_bins: int = DEFAULT_BINS) -> "ScoreHist":
        """``typology`` holds class indices (0 = legitimate, k > 0 = typology k)."""
        bins = score_bins(scores, n_bins)
        flat = np.asarray(typology, dtype=np.int64) * n_bins + bins
        counts = np.bincount(flat, minlength=N_CLASSES * n_bins).reshape(N_CLASSES, n_bins)
        return cls(counts.astype(np.int64))

    def __add__(self, other: "ScoreHist") -> "ScoreHist":
        return ScoreHist(self.counts + other.counts)

    def to_list(self) -> list[int]:
        return [int(v) for v in self.counts.ravel()]

    @classmethod
    def from_list(cls, values: list[int] | np.ndarray, n_bins: int | None = None) -> "ScoreHist":
        arr = np.rint(np.asarray(values, dtype=np.float64)).astype(np.int64)
        n_bins = n_bins or arr.size // N_CLASSES
        return cls(arr.reshape(N_CLASSES, n_bins))

    # ------------------------------------------------------------------ basic counts
    @property
    def n_bins(self) -> int:
        return self.counts.shape[1]

    @property
    def n(self) -> int:
        return int(self.counts.sum())

    @property
    def n_pos(self) -> int:
        return int(self.counts[1:].sum())

    @property
    def n_neg(self) -> int:
        return int(self.counts[0].sum())

    def _tails(self) -> tuple[np.ndarray, np.ndarray]:
        """tp[k], fp[k] when flagging bins >= k, for k = 0..n_bins."""
        pos = self.counts[1:].sum(axis=0)
        neg = self.counts[0]
        tp = np.concatenate([np.cumsum(pos[::-1])[::-1], [0]])
        fp = np.concatenate([np.cumsum(neg[::-1])[::-1], [0]])
        return tp.astype(np.float64), fp.astype(np.float64)

    # ------------------------------------------------------------------ metrics
    def best_threshold(self, metric: str = "f1") -> int:
        """Bin edge that maximises ``metric`` (ties: the higher threshold)."""
        tp, fp = self._tails()
        fn = self.n_pos - tp
        if metric == "f1":
            value = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
        elif metric == "f2":
            value = np.divide(5 * tp, 5 * tp + 4 * fn + fp, out=np.zeros_like(tp), where=(5 * tp + 4 * fn + fp) > 0)
        else:
            raise ValueError(f"unknown threshold metric '{metric}'")
        best = np.flatnonzero(value == value.max())
        return int(best[-1])

    def at(self, k: int) -> dict[str, Any]:
        """Confusion counts and derived metrics when flagging bins >= k."""
        k = int(np.clip(k, 0, self.n_bins))
        flagged = self.counts[:, k:].sum(axis=1)
        tp = int(flagged[1:].sum())
        fp = int(flagged[0])
        fn = self.n_pos - tp
        tn = self.n_neg - fp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class = self.counts[1:].sum(axis=1)
        recall_by_typology = {
            TYPOLOGY_CLASSES[c]: (float(flagged[c] / per_class[c - 1]) if per_class[c - 1] else None) for c in range(1, N_CLASSES)
        }
        return {
            "threshold": bin_threshold(k, self.n_bins),
            "threshold_bin": k,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "alert_rate": (tp + fp) / self.n if self.n else 0.0,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "recall_by_typology": recall_by_typology,
            "support_by_typology": {TYPOLOGY_CLASSES[c]: int(per_class[c - 1]) for c in range(1, N_CLASSES)},
        }

    def pr_auc(self) -> float:
        """Average precision: sum of precision at each threshold times the recall gained."""
        if self.n_pos == 0:
            return float("nan")
        tp, fp = self._tails()
        tp, fp = tp[::-1], fp[::-1]  # from the strictest threshold to the loosest
        precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)
        recall = tp / self.n_pos
        return float(np.sum(np.diff(recall) * precision[1:]))

    def roc_auc(self) -> float:
        """Area under the ROC curve (trapezoid: ties inside a bin count one half)."""
        if self.n_pos == 0 or self.n_neg == 0:
            return float("nan")
        tp, fp = self._tails()
        tpr = tp[::-1] / self.n_pos
        fpr = fp[::-1] / self.n_neg
        return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))

    def summary(self, k: int) -> dict[str, Any]:
        """All reported metrics at threshold bin ``k``."""
        out = self.at(k)
        out.update({"pr_auc": self.pr_auc(), "roc_auc": self.roc_auc(), "n": self.n, "n_illicit": self.n_pos,
                    "prevalence": self.n_pos / self.n if self.n else 0.0})
        return out


def evaluate_hists(val: ScoreHist, test: ScoreHist, metric: str = "f1") -> dict[str, Any]:
    """Tune the decision threshold on validation, report on test.

    The test labels never influence the threshold. ``oracle_f1`` (the best F1
    any threshold could reach on test) is reported only to show how much is lost
    by transferring the validation threshold.
    """
    k = val.best_threshold(metric)
    result = test.summary(k)
    result["val"] = {key: val.summary(k)[key] for key in ["precision", "recall", "f1", "pr_auc", "roc_auc", "n", "n_illicit"]}
    result["oracle_f1"] = test.at(test.best_threshold("f1"))["f1"]
    return result


def evaluate_scores(
    val_scores: np.ndarray,
    val_typology: np.ndarray,
    test_scores: np.ndarray,
    test_typology: np.ndarray,
    metric: str = "f1",
    n_bins: int = DEFAULT_BINS,
) -> dict[str, Any]:
    """Convenience wrapper: histogram both splits, then :func:`evaluate_hists`."""
    return evaluate_hists(ScoreHist.from_scores(val_scores, val_typology, n_bins), ScoreHist.from_scores(test_scores, test_typology, n_bins), metric)
