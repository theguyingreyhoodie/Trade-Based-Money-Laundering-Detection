"""Calibrated risk score (Step 05, objective 5).

    risk = 100 * C( w * P_model + (1 - w) * P_rules )

* ``P_model``  the global FL model's output, calibrated to a probability;
* ``P_rules``  the rules engine's noisy-OR score, calibrated to a probability;
* ``w``        ``scoring.model_weight``;
* ``C``        a final calibration of the blend, so that a score of 70 means
               "about 70% of trades scored like this were illicit on validation".

Every calibration map is fitted on **validation** data and only from score
histograms, the aggregate that FL clients already share for evaluation. Fitting
the scorer therefore needs no pooled predictions: clients send three more
histograms (rule scores, blended scores, and the model scores they sent anyway).

Why blend at all (the weighting is discussed in ``docs/scoring.md``):

* the model ranks far better than the rules, so it carries most of the weight;
* the rules are transparent, encode recognised red flags, and catch the
  cross-document schemes deterministically, so they keep a material weight;
* rule evidence is partly independent of the model, so the blend ranks better
  than either part alone.

**Rule floor.** Because the blend is recalibrated, a rule hit that the model
disagrees with ends up with a low score: on validation such trades are mostly
legitimate. That is the right probability, but it means a blind, drifting or
poisoned model could keep a trade with strong red flags out of review. The
scorer therefore applies an explicit policy floor: a trade whose calibrated rule
probability alone is at least ``scoring.rule_floor_probability`` is scored at
least at the Medium threshold. The floor is a review policy, not a probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tbml.eval.metrics import LOGIT_RANGE, ScoreHist

BANDS = ["Low", "Medium", "High", "Critical"]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    with np.errstate(divide="ignore"):
        x = np.log(p) - np.log1p(-p)
    return np.clip(np.nan_to_num(x, posinf=LOGIT_RANGE, neginf=-LOGIT_RANGE), -LOGIT_RANGE, LOGIT_RANGE)


def bin_centres(n_bins: int) -> np.ndarray:
    """Centre of each score bin on the logit scale."""
    edges = np.linspace(-LOGIT_RANGE, LOGIT_RANGE, n_bins + 1)
    return (edges[:-1] + edges[1:]) / 2.0


@dataclass
class Calibrator:
    """Monotone map from a raw score in [0, 1] to a probability, stored as knots on the logit scale."""

    method: str
    x: list[float]  # knots: logit of the raw score
    y: list[float]  # calibrated probability at each knot

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        return np.interp(_logit(scores), self.x, self.y)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "x": [float(v) for v in self.x], "y": [float(v) for v in self.y]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Calibrator":
        return cls(d["method"], list(d["x"]), list(d["y"]))


def identity_calibrator() -> Calibrator:
    """Calibrator that returns its input unchanged."""
    grid = np.linspace(-LOGIT_RANGE, LOGIT_RANGE, 961)
    return Calibrator("identity", list(grid), list(1.0 / (1.0 + np.exp(-grid))))


def fit_calibrator(hist: ScoreHist, method: str = "isotonic") -> Calibrator:
    """Fit a calibration map from a validation score histogram.

    ``isotonic``: weighted isotonic regression of the positive rate per bin.
    ``platt``: weighted logistic regression on the logit of the score.
    """
    pos = hist.counts[1:].sum(axis=0).astype(np.float64)
    total = hist.counts.sum(axis=0).astype(np.float64)
    centres = bin_centres(hist.n_bins)
    filled = total > 0
    if method not in ("isotonic", "platt"):
        raise ValueError(f"unknown calibration method '{method}' (isotonic | platt)")
    if not filled.any():
        return identity_calibrator()
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        iso.fit(centres[filled], pos[filled] / total[filled], sample_weight=total[filled])
        y = iso.predict(centres[filled])
        return Calibrator("isotonic", list(centres[filled]), list(y))
    else:
        from sklearn.linear_model import LogisticRegression

        x = np.concatenate([centres[filled], centres[filled]])
        y = np.concatenate([np.ones(int(filled.sum())), np.zeros(int(filled.sum()))])
        w = np.concatenate([pos[filled], (total - pos)[filled]])
        keep = w > 0
        if len(np.unique(y[keep])) < 2:
            rate = float(pos.sum() / total.sum())
            return Calibrator("platt", [-LOGIT_RANGE, LOGIT_RANGE], [rate, rate])
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(x[keep, None], y[keep], sample_weight=w[keep])
        grid = np.linspace(-LOGIT_RANGE, LOGIT_RANGE, 241)
        return Calibrator("platt", list(grid), list(lr.predict_proba(grid[:, None])[:, 1]))


def calibration_report(hist: ScoreHist, calibrator: Calibrator | None = None, n_groups: int = 10) -> dict[str, Any]:
    """Brier score and expected calibration error of (optionally calibrated) scores, from a histogram."""
    pos = hist.counts[1:].sum(axis=0).astype(np.float64)
    total = hist.counts.sum(axis=0).astype(np.float64)
    raw = 1.0 / (1.0 + np.exp(-bin_centres(hist.n_bins)))
    p = calibrator(raw) if calibrator is not None else raw
    n = total.sum()
    if n == 0:
        return {"brier": None, "ece": None, "reliability": []}
    brier = float((pos * (1 - p) ** 2 + (total - pos) * p**2).sum() / n)
    groups = np.minimum((p * n_groups).astype(int), n_groups - 1)
    ece, table = 0.0, []
    for g in range(n_groups):
        m = groups == g
        if total[m].sum() == 0:
            continue
        predicted = float((p[m] * total[m]).sum() / total[m].sum())
        observed = float(pos[m].sum() / total[m].sum())
        ece += abs(predicted - observed) * total[m].sum() / n
        table.append({"bucket": f"{g / n_groups:.1f}-{(g + 1) / n_groups:.1f}", "n": int(total[m].sum()), "mean_predicted": predicted, "observed_rate": observed})
    return {"brier": brier, "ece": float(ece), "reliability": table}


@dataclass
class RiskScorer:
    """Turns a model probability and a rule score into a 0-100 risk score and a band."""

    model_weight: float
    model_calibrator: Calibrator
    rules_calibrator: Calibrator
    final_calibrator: Calibrator
    bands: dict[str, float] = field(default_factory=lambda: {"medium": 25.0, "high": 50.0, "critical": 75.0})
    rule_floor_probability: float = 1.1  # above 1 switches the floor off

    def components(self, model_score: np.ndarray, rule_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calibrated probabilities of the two components."""
        return self.model_calibrator(model_score), self.rules_calibrator(rule_score)

    def blend(self, model_score: np.ndarray, rule_score: np.ndarray) -> np.ndarray:
        p_model, p_rules = self.components(model_score, rule_score)
        return self.model_weight * p_model + (1.0 - self.model_weight) * p_rules

    def calibrated_score(self, model_score: np.ndarray, rule_score: np.ndarray) -> np.ndarray:
        """The calibrated blend on a 0-100 scale, before the rule floor."""
        return 100.0 * self.final_calibrator(self.blend(model_score, rule_score))

    def floored(self, model_score: np.ndarray, rule_score: np.ndarray) -> np.ndarray:
        """True where the rule floor, not the calibrated blend, decides the score."""
        strong = self.rules_calibrator(rule_score) >= self.rule_floor_probability
        return strong & (self.calibrated_score(model_score, rule_score) < self.bands["medium"])

    def score(self, model_score: np.ndarray, rule_score: np.ndarray) -> np.ndarray:
        """Risk score in [0, 100]: the calibrated blend, raised to Medium where the rule floor applies."""
        risk = self.calibrated_score(model_score, rule_score)
        strong = self.rules_calibrator(rule_score) >= self.rule_floor_probability
        return np.where(strong, np.maximum(risk, self.bands["medium"]), risk)

    def band(self, risk: np.ndarray) -> np.ndarray:
        risk = np.asarray(risk, dtype=np.float64)
        idx = (risk >= self.bands["medium"]).astype(int) + (risk >= self.bands["high"]) + (risk >= self.bands["critical"])
        return np.array(BANDS, dtype=object)[idx]

    def to_dict(self) -> dict[str, Any]:
        return {"model_weight": self.model_weight, "bands": dict(self.bands), "rule_floor_probability": self.rule_floor_probability,
                "model_calibrator": self.model_calibrator.to_dict(),
                "rules_calibrator": self.rules_calibrator.to_dict(), "final_calibrator": self.final_calibrator.to_dict()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RiskScorer":
        return cls(float(d["model_weight"]), Calibrator.from_dict(d["model_calibrator"]), Calibrator.from_dict(d["rules_calibrator"]),
                   Calibrator.from_dict(d["final_calibrator"]), dict(d["bands"]), float(d.get("rule_floor_probability", 1.1)))


def fit_risk_scorer(
    val_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    model_weight: float,
    method: str = "isotonic",
    bands: dict[str, float] | None = None,
    n_bins: int = 800,
    rule_floor_probability: float = 1.1,
) -> tuple[RiskScorer, dict[str, ScoreHist]]:
    """Fit all calibration maps from per-client validation histograms.

    ``val_parts`` holds, per client, ``(model_score, rule_score, typology)`` of
    its validation trades. Only histograms of these are summed across clients,
    mirroring what a deployment would exchange:

    1. histogram of model scores  -> model calibrator
    2. histogram of rule scores   -> rules calibrator
    3. histogram of the blend     -> final calibrator
    """
    h_model = sum((ScoreHist.from_scores(m, t, n_bins) for m, _, t in val_parts), ScoreHist.empty(n_bins))
    h_rules = sum((ScoreHist.from_scores(r, t, n_bins) for _, r, t in val_parts), ScoreHist.empty(n_bins))
    model_cal = fit_calibrator(h_model, method)
    rules_cal = fit_calibrator(h_rules, method)
    scorer = RiskScorer(float(model_weight), model_cal, rules_cal, identity_calibrator(), dict(bands or {"medium": 25.0, "high": 50.0, "critical": 75.0}),
                        float(rule_floor_probability))
    h_blend = sum((ScoreHist.from_scores(scorer.blend(m, r), t, n_bins) for m, r, t in val_parts), ScoreHist.empty(n_bins))
    scorer.final_calibrator = fit_calibrator(h_blend, method)
    return scorer, {"model": h_model, "rules": h_rules, "blend": h_blend}


def weight_sensitivity(val_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]], scorer: RiskScorer, weights: list[float], n_bins: int = 800) -> list[dict[str, float]]:
    """Validation PR-AUC and best F1 of the blend for each candidate model weight."""
    rows = []
    for w in weights:
        trial = RiskScorer(float(w), scorer.model_calibrator, scorer.rules_calibrator, scorer.final_calibrator, scorer.bands, scorer.rule_floor_probability)
        hist = sum((ScoreHist.from_scores(trial.blend(m, r), t, n_bins) for m, r, t in val_parts), ScoreHist.empty(n_bins))
        rows.append({"model_weight": float(w), "val_pr_auc": hist.pr_auc(), "val_best_f1": hist.at(hist.best_threshold("f1"))["f1"]})
    return rows
