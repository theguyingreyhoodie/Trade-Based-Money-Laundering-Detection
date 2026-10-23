"""SHAP explanations for a bank's flagged transactions, and the alert records built from them.

Each bank explains its own alerts with its own model and its own data (shap.DeepExplainer
against a background sample of its training rows). Contributions are in log-odds of the
laundering score: positive values push a transaction towards an alert. One-hot columns are
summed back to the column they came from, so "payment_format" shows as one feature with the
transaction's actual value.

An alert's hash covers txn_id, bank, score, top features and model_version; that hash is what
goes on-chain (chain.chain_client.AuditChain.log_alert), never the alert itself.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
from torch import nn


class _Column(nn.Module):
    """SHAP expects a (n, 1) output; our models return (n,) logits."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).unsqueeze(1)


def shap_values(model: nn.Module, background: np.ndarray, X: np.ndarray,
                batch_size: int = 2048) -> tuple[np.ndarray, float]:
    """Per-feature contributions (n, d) to the logit, and the base value (mean background logit).
    For every row, base + contributions.sum() equals the model's logit."""
    import shap

    model.eval()
    wrapped = _Column(model)
    bg = torch.from_numpy(np.ascontiguousarray(background, dtype=np.float32))
    explainer = shap.DeepExplainer(wrapped, bg)
    parts = [np.asarray(explainer.shap_values(torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size],
                                                                                   dtype=np.float32))))
             .reshape(min(batch_size, len(X) - i), -1) for i in range(0, len(X), batch_size)]
    with torch.no_grad():
        base = float(wrapped(bg).mean())
    return (np.vstack(parts) if parts else np.zeros((0, X.shape[1]))), base


def column_of(feature_name: str) -> str:
    """'payment_format=ACH' -> 'payment_format'; 'kyc_risk_level=<missing>' -> 'kyc_risk_level'."""
    return feature_name.split("=", 1)[0]


def group_by_column(contrib: np.ndarray, feature_names: list[str]) -> tuple[list[str], np.ndarray]:
    """Sum the contributions of each column's one-hot features: (n, d) -> (n, columns)."""
    columns = list(dict.fromkeys(column_of(f) for f in feature_names))
    index = {c: i for i, c in enumerate(columns)}
    out = np.zeros((contrib.shape[0], len(columns)))
    for j, f in enumerate(feature_names):
        out[:, index[column_of(f)]] += contrib[:, j]
    return columns, out


def top_features(columns: list[str], row: np.ndarray, values: dict, k: int = 5) -> list[dict]:
    """The k columns with the largest absolute contribution, with the transaction's raw value."""
    order = np.argsort(-np.abs(row))[:k]
    return [{"feature": columns[i], "value": _plain(values.get(columns[i])),
             "contribution": round(float(row[i]), 6)} for i in order]


def _plain(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v


def alert_hash(alert: dict) -> str:
    """SHA-256 over the fields that define an alert. Recomputing it from a stored alert and
    comparing with the chain shows whether the alert was changed after it was logged."""
    core = {"txn_id": int(alert["txn_id"]), "bank": alert["bank"], "score": round(float(alert["score"]), 6),
            "top_features": [[f["feature"], round(float(f["contribution"]), 6)] for f in alert["top_features"]],
            "model_version": alert["model_version"]}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
