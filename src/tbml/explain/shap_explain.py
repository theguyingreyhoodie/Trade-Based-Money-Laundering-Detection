"""SHAP attributions for the global model.

Attributions are computed on the model's logit (log-odds of "illicit"), where
they are additive: base value + sum of contributions = the trade's logit. The
background sample is drawn from the explaining institution's own training rows,
so an explanation needs no data from anyone else.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
from torch import nn

from tbml.explain.phrases import describe


class _LogitColumn(nn.Module):
    """Wrap the model so its output has shape [n, 1], which SHAP's deep explainer expects."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).unsqueeze(-1)


@torch.no_grad()
def model_logits(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    return model(torch.as_tensor(np.asarray(X, dtype=np.float32))).numpy().astype(np.float64)


def explain(model: nn.Module, background: np.ndarray, X: np.ndarray, tolerance: float = 0.05, permutations: int = 20, seed: int = 0) -> dict[str, Any]:
    """SHAP values of ``X`` (shape [n, d], logit scale) against ``background``.

    DeepSHAP is tried first. Its contributions must add up to the model output
    within ``tolerance``; otherwise the model-agnostic permutation explainer is
    used, which is slower but exact in that respect. In this project the
    fallback is the rule rather than the exception: the model clips standardised
    inputs at +/- 8, alerted trades often have an indicator beyond that (which
    is why they are alerted), and DeepSHAP's linear rule is inaccurate across
    the clip. The method and the additivity error are returned and reported.

    The permutation explainer averages over ``permutations`` random feature
    orderings (each evaluated forwards and backwards), seeded with ``seed`` so
    explanations are reproducible. SHAP's default budget (500 evaluations)
    allows only 3 orderings for 81 indicators; on alerted trades of the primary
    run that left 93% of the top-5 indicators in agreement with a 100-ordering
    reference, against 97% with 20 orderings (see ``docs/scoring.md``).
    """
    import shap

    model.eval()
    X = np.asarray(X, dtype=np.float32)
    background = np.asarray(background, dtype=np.float32)
    base = float(model_logits(model, background).mean())
    if len(X) == 0:
        return {"values": np.zeros((0, background.shape[1])), "base_value": base, "method": "none", "additivity_error": 0.0}
    target = model_logits(model, X)
    method = "deep"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            explainer = shap.DeepExplainer(_LogitColumn(model), torch.as_tensor(background))
            raw = explainer.shap_values(torch.as_tensor(X), check_additivity=False)
            values = np.asarray(raw[0] if isinstance(raw, list) else raw, dtype=np.float64).reshape(len(X), -1)
            error = float(np.abs(values.sum(axis=1) + base - target).max())
        except Exception:  # fall through to the model-agnostic explainer
            values, error = None, float("inf")
        if values is None or error > tolerance:
            method = "permutation"
            fn = lambda a: model_logits(model, a)  # noqa: E731
            masker = shap.maskers.Independent(background, max_samples=len(background))
            budget = int(permutations) * (2 * X.shape[1] + 1)
            explanation = shap.PermutationExplainer(fn, masker, seed=int(seed) % (2**32))(X, max_evals=budget, silent=True)
            values = np.asarray(explanation.values, dtype=np.float64)
            base = float(np.mean(explanation.base_values))
            error = float(np.abs(values.sum(axis=1) + base - target).max())
    return {"values": values, "base_value": base, "method": method, "additivity_error": error, "permutations": int(permutations) if method == "permutation" else None}


def top_contributions(shap_row: np.ndarray, x_row: np.ndarray, names: list[str], k: int) -> list[dict[str, Any]]:
    """The ``k`` indicators with the largest absolute contribution, with plain-English wording."""
    order = np.argsort(-np.abs(shap_row))[:k]
    return [
        {
            "feature": names[j],
            "value": round(float(x_row[j]), 6),
            "shap": round(float(shap_row[j]), 4),
            "direction": "raises risk" if shap_row[j] > 0 else "lowers risk",
            "text": describe(names[j], float(x_row[j])),
        }
        for j in order
        if shap_row[j] != 0
    ]
