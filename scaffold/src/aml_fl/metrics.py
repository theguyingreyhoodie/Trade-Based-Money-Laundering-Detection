"""Evaluation metrics for heavily imbalanced detection. PR-AUC is the headline metric."""
from __future__ import annotations

import numpy as np
import pandas as pd
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


def pooled_evaluate(y_true: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> dict[str, float]:
    """Metrics over several banks' test sets together, each row flagged by its OWN bank's threshold.

    Threshold metrics (precision, recall, f1, fpr) use `score >= threshold` per row. PR-AUC and
    ROC-AUC rank the raw scores of all banks together.
    """
    scores, thresholds = np.asarray(scores, dtype=float), np.asarray(thresholds, dtype=float)
    out = evaluate(y_true, scores - thresholds, 0.0)
    raw = evaluate(y_true, scores)
    out.update(pr_auc=raw["pr_auc"], roc_auc=raw["roc_auc"], threshold=float("nan"))
    return out


def recall_by_pattern(patterns: np.ndarray, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    """Recall per laundering pattern (see patterns.py). Non-laundering rows are ignored.

    Columns: pattern, positives, detected, recall. Rows sorted by number of positives.
    """
    patterns = np.asarray(patterns)
    mask = patterns != "none"
    df = pd.DataFrame({"pattern": patterns[mask], "detected": np.asarray(scores)[mask] >= threshold})
    out = df.groupby("pattern")["detected"].agg(positives="size", detected="sum").reset_index()
    out["detected"] = out["detected"].astype(int)
    out["recall"] = out["detected"] / out["positives"]
    return out.sort_values("positives", ascending=False, ignore_index=True)
