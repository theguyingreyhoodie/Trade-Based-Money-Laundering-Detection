"""Evaluation metrics for heavily imbalanced detection. PR-AUC is the headline metric."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Pick the threshold that maximizes F1 — use on the VALIDATION split only."""
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(t[np.argmax(f1[:-1])]) if len(t) else 0.5


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    pred = (scores >= threshold).astype(int)
    has_both = 0 < y_true.sum() < len(y_true)
    neg = max(1, int((y_true == 0).sum()))
    return {
        "pr_auc": float(average_precision_score(y_true, scores)) if has_both else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, scores)) if has_both else float("nan"),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "fpr": float(((pred == 1) & (y_true == 0)).sum() / neg),
        "threshold": float(threshold),
        "positives": int(y_true.sum()),
        "n": int(len(y_true)),
    }
