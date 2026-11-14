"""Risk scoring, calibration, SHAP explanations and alerts."""

from __future__ import annotations

import copy
import re

import numpy as np
import pytest
import torch

from tbml.data.typologies import TYPOLOGIES
from tbml.eval.metrics import N_CLASSES, ScoreHist
from tbml.explain.alerts import alert_digest, alert_markdown, register_alerts, run_alert_pipeline, sample_alerts, suspected_typology, verify_alert
from tbml.explain.phrases import PHRASES, describe
from tbml.explain.scoring import BANDS, Calibrator, RiskScorer, calibration_report, fit_calibrator, fit_risk_scorer, weight_sensitivity
from tbml.explain.shap_explain import explain, model_logits, top_contributions
from tbml.features.dictionary import FEATURE_INDEX, FEATURE_NAMES
from tbml.fl.data import load_client, prepare_clients, scenario_dir
from tbml.fl.partition import Scenario
from tbml.ledger.factory import open_ledger
from tbml.models.mlp import feature_moments, fit_mlp, make_model, moments_to_scaler, set_scaler
from tbml.rules.engine import RULE_IDS, RULES, rule_severities

# --------------------------------------------------------------------------- calibration


def _miscalibrated(n: int = 40000, seed: int = 0):
    """Scores that rank well but are over-confident (true probability = score ** 2)."""
    rng = np.random.default_rng(seed)
    s = rng.beta(0.6, 3.0, n)
    y = rng.random(n) < s**2
    typ = np.where(y, rng.integers(1, N_CLASSES, n), 0)
    return s, typ


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_calibration_improves_reliability(method: str) -> None:
    s_val, t_val = _miscalibrated(seed=0)
    s_test, t_test = _miscalibrated(seed=1)
    cal = fit_calibrator(ScoreHist.from_scores(s_val, t_val), method)
    test_hist = ScoreHist.from_scores(s_test, t_test)
    before, after = calibration_report(test_hist), calibration_report(test_hist, cal)
    assert after["ece"] < before["ece"] / 2 and after["brier"] < before["brier"]
    grid = np.linspace(0.0, 1.0, 400)
    out = cal(grid)
    assert (np.diff(out) >= -1e-12).all(), "a calibration map must never reverse the ranking"
    assert out.min() >= 0.0 and out.max() <= 1.0
    if method == "isotonic":
        assert abs(cal(np.array([0.5]))[0] - 0.25) < 0.05, "true probability at score 0.5 is 0.25"
    restored = Calibrator.from_dict(cal.to_dict())
    np.testing.assert_allclose(restored(grid), out)


def test_calibrator_handles_degenerate_histograms() -> None:
    empty = fit_calibrator(ScoreHist.empty())
    np.testing.assert_allclose(empty(np.array([0.01, 0.3, 0.9])), [0.01, 0.3, 0.9], atol=0.002)
    no_pos = fit_calibrator(ScoreHist.from_scores(np.array([0.2, 0.4, 0.6]), np.zeros(3, dtype=int)))
    assert (no_pos(np.array([0.1, 0.9])) == 0.0).all()
    with pytest.raises(ValueError):
        fit_calibrator(ScoreHist.empty(), "magic")


# --------------------------------------------------------------------------- risk score


def _val_parts(n_clients: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(n_clients):
        n = 6000
        y = rng.random(n) < 0.05
        model = 1 / (1 + np.exp(-(rng.normal(-3.5, 1.2, n) + 4.0 * y)))
        rules = np.where(rng.random(n) < np.where(y, 0.55, 0.04), rng.choice([0.4, 0.6, 0.84, 0.9], n), 0.0)
        parts.append((model, rules, np.where(y, rng.integers(1, N_CLASSES, n), 0)))
    return parts


def test_risk_score_range_bands_and_monotonicity() -> None:
    parts = _val_parts()
    scorer, hists = fit_risk_scorer(parts, model_weight=0.7)
    assert set(hists) == {"model", "rules", "blend"} and hists["model"].n == 18000
    m = np.linspace(0, 1, 60)
    r = np.linspace(0, 0.99, 60)
    grid = scorer.score(*np.meshgrid(m, r))
    assert grid.min() >= 0.0 and grid.max() <= 100.0
    assert (np.diff(grid, axis=1) >= -1e-9).all(), "a higher model probability never lowers the risk score"
    assert (np.diff(grid, axis=0) >= -1e-9).all(), "more rule evidence never lowers the risk score"
    assert scorer.score(np.array([0.999]), np.array([0.9]))[0] > 75 > 25 > scorer.score(np.array([0.001]), np.array([0.0]))[0]
    assert list(scorer.band(np.array([0.0, 24.9, 25.0, 49.9, 50.0, 74.9, 75.0, 100.0]))) == ["Low", "Low", "Medium", "Medium", "High", "High", "Critical", "Critical"]
    assert BANDS == ["Low", "Medium", "High", "Critical"]
    clone = RiskScorer.from_dict(scorer.to_dict())
    np.testing.assert_allclose(clone.score(m, r), scorer.score(m, r))


def test_rule_floor_keeps_strong_red_flags_in_review() -> None:
    """Strong rule evidence puts a trade in at least the Medium band even when the model sees nothing."""
    from tbml.explain.scoring import identity_calibrator

    # a final calibration that scores "model silent, rule fired" low, as it does when such trades are mostly legitimate
    grid = np.linspace(-12, 12, 481)
    p = 1 / (1 + np.exp(-grid))
    pessimistic = Calibrator("isotonic", list(grid), list(p**3))
    bands = {"medium": 25.0, "high": 50.0, "critical": 75.0}
    without = RiskScorer(0.7, identity_calibrator(), identity_calibrator(), pessimistic, bands)
    with_floor = RiskScorer(0.7, identity_calibrator(), identity_calibrator(), pessimistic, bands, rule_floor_probability=0.6)

    quiet, strong, weak = np.array([0.001]), np.array([0.9]), np.array([0.4])
    assert without.score(quiet, strong)[0] < 5.0, "without the floor the red flag is scored away"
    assert with_floor.score(quiet, strong)[0] == 25.0 and with_floor.band(with_floor.score(quiet, strong))[0] == "Medium"
    assert with_floor.floored(quiet, strong)[0] and not without.floored(quiet, strong)[0]
    assert with_floor.score(quiet, weak)[0] < 5.0, "weak rule evidence does not trigger the floor"
    assert not with_floor.floored(np.array([0.99]), strong)[0], "the floor is idle when the blend is already above Medium"

    rng = np.random.default_rng(1)
    m, r = rng.random(2000), rng.choice([0.0, 0.4, 0.9], 2000)
    assert (with_floor.score(m, r) >= without.score(m, r)).all(), "the floor never lowers a score"
    acted = with_floor.floored(m, r)
    assert acted.any() and (with_floor.score(m, r)[acted] == 25.0).all() and (r[acted] == 0.9).all()
    np.testing.assert_allclose(with_floor.score(m, r)[~acted], without.score(m, r)[~acted])
    np.testing.assert_allclose(without.score(m, r), without.calibrated_score(m, r))
    assert RiskScorer.from_dict(with_floor.to_dict()).rule_floor_probability == 0.6
    # fitted from data, the default is "floor off"
    fitted, _ = fit_risk_scorer(_val_parts(), model_weight=0.7)
    assert not fitted.floored(m, r).any()


def test_score_is_calibrated_and_blend_helps() -> None:
    val, test = _val_parts(seed=0), _val_parts(seed=1)
    scorer, _ = fit_risk_scorer(val, model_weight=0.7)
    risk = np.concatenate([scorer.score(m, r) for m, r, _ in test])
    y = np.concatenate([t > 0 for _, _, t in test])
    for lo, hi in [(25, 50), (50, 75), (75, 100.01)]:
        band = (risk >= lo) & (risk < hi)
        if band.sum() >= 50:
            assert lo - 15 <= 100 * y[band].mean() <= hi + 15, (lo, hi, y[band].mean())
    rows = {row["model_weight"]: row for row in weight_sensitivity(val, scorer, [0.0, 0.7, 1.0])}
    assert rows[0.7]["val_pr_auc"] > rows[1.0]["val_pr_auc"] > rows[0.0]["val_pr_auc"], "independent rule evidence must add to the model"


# --------------------------------------------------------------------------- plain-English phrases


def test_every_feature_has_a_phrase() -> None:
    assert set(PHRASES) == set(FEATURE_NAMES)
    for name in FEATURE_NAMES:
        for value in (0.0, 1.0, -1.5, 2.3, 7.0):
            text = describe(name, value)
            assert isinstance(text, str) and len(text) > 8 and not re.search(r"\b(nan|inf)\b", text.lower()), (name, value, text)
            assert name not in text, "a phrase must not expose the raw column name"


def test_phrases_read_naturally() -> None:
    assert describe("price_log_ratio", np.log(3.4)) == "unit price 3.4x the HS reference price"
    assert describe("price_log_ratio", 0.0) == "unit price about equal to the HS reference price"
    assert describe("missing_both", 1.0) == "neither a shipment record nor a customs declaration exists"
    assert describe("invoices_per_bl", 3.0) == "3 invoices quote the same bill of lading"
    assert describe("min_party_age_log", np.log1p(120)) == "younger party registered only 4 months before the invoice"
    assert describe("n_transshipments", 1.0) == "1 transshipment stop" and describe("n_transshipments", 2.0) == "2 transshipment stops"
    assert describe("days_invoice_to_ship", -6.0) == "shipped 6 days before the invoice was issued"
    assert describe("days_payment_minus_ship", -12.0) == "first payment 12 days before departure"
    assert describe("customs_weight_per_unit_log_ratio", np.log(0.004)).startswith("declared weight per unit is 0.004x")
    assert describe("max_addr_shared_log", np.log1p(1)) == "a party shares its address with 1 other known entity"
    assert describe("route_detour_ratio", 3.2) == "route is 3.2x the direct distance"
    assert "no_such_feature" in describe("no_such_feature", 1.0)


# --------------------------------------------------------------------------- SHAP


def test_shap_values_are_additive(cfg) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1200, 8)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 1.0).astype(np.int8)
    model, _ = fit_mlp(cfg, X[:1000], y[:1000], X[1000:1100], y[1000:1100], seed=0, epochs=15)
    X = X[600:]
    result = explain(model, X[:100], X[500:520])
    assert result["values"].shape == (20, 8) and result["method"] in {"deep", "permutation"}
    assert result["additivity_error"] < 0.05
    np.testing.assert_allclose(result["values"].sum(axis=1) + result["base_value"], model_logits(model, X[500:520]), atol=0.05)
    # the informative features carry the attribution
    importance = np.abs(result["values"]).mean(axis=0)
    assert importance.argmax() == 0 and importance[:2].sum() > importance[2:].sum()
    assert explain(model, X[:100], X[:0])["values"].shape == (0, 8)


def test_permutation_shap_is_reproducible(cfg) -> None:
    """With a fixed seed, explanations from the permutation explainer are identical between runs."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 6)).astype(np.float32)
    model = make_model(cfg, 6, seed=4)
    with torch.no_grad():
        set_scaler(model, *moments_to_scaler(*feature_moments(X)))
    extreme = X[:8].copy()
    extreme[:, 0] = 60.0  # beyond the clip: forces the permutation explainer
    a = explain(model, X[:100], extreme, permutations=5, seed=11)
    b = explain(model, X[:100], extreme, permutations=5, seed=11)
    c = explain(model, X[:100], extreme, permutations=5, seed=12)
    assert a["method"] == "permutation" and a["permutations"] == 5
    np.testing.assert_array_equal(a["values"], b["values"])
    assert not np.array_equal(a["values"], c["values"])
    assert max(a["additivity_error"], c["additivity_error"]) < 1e-4


def test_shap_falls_back_when_inputs_are_clipped(cfg) -> None:
    """Beyond the +/-8 clip DeepSHAP is inaccurate; the result must still be additive."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 6)).astype(np.float32)
    model = make_model(cfg, 6, seed=3)
    with torch.no_grad():
        set_scaler(model, *moments_to_scaler(*feature_moments(X)))
    extreme = X[:10].copy()
    extreme[:, 0] = 60.0
    result = explain(model, X[:100], extreme)
    assert result["additivity_error"] < 0.05


def test_top_contributions() -> None:
    names = FEATURE_NAMES
    shap_row = np.zeros(len(names))
    x = np.zeros(len(names), dtype=np.float32)
    shap_row[FEATURE_INDEX["price_log_ratio"]], x[FEATURE_INDEX["price_log_ratio"]] = 2.5, np.log(3.0)
    shap_row[FEATURE_INDEX["payer_related"]], x[FEATURE_INDEX["payer_related"]] = -1.2, 1.0
    shap_row[FEATURE_INDEX["n_payments"]], x[FEATURE_INDEX["n_payments"]] = 0.3, 2.0
    top = top_contributions(shap_row, x, names, k=2)
    assert [t["feature"] for t in top] == ["price_log_ratio", "payer_related"]
    assert top[0]["direction"] == "raises risk" and top[1]["direction"] == "lowers risk"
    assert top[0]["text"] == "unit price 3.0x the HS reference price"
    assert len(top_contributions(shap_row, x, names, k=10)) == 3, "indicators without a contribution are not listed"


# --------------------------------------------------------------------------- suspected typology


def test_suspected_typology_combines_model_and_rules(cfg) -> None:
    severities = rule_severities(cfg)
    none = np.zeros(len(RULES), dtype=bool)
    shap_row = np.zeros(len(FEATURE_NAMES))
    shap_row[FEATURE_INDEX["route_detour_ratio"]] = 3.0
    shap_row[FEATURE_INDEX["route_via_high_risk"]] = 1.0
    typology, scores = suspected_typology(shap_row, FEATURE_NAMES, none, severities)
    assert typology == "high_risk_routing" and scores["high_risk_routing"] == pytest.approx(1.0)

    hits = none.copy()
    hits[RULE_IDS.index("R04")] = True  # several invoices exceed one shipment
    typology, scores = suspected_typology(np.zeros(len(FEATURE_NAMES)), FEATURE_NAMES, hits, severities)
    assert typology == "multiple_invoicing"

    typology, scores = suspected_typology(shap_row, FEATURE_NAMES, hits, severities)
    assert set(list(scores)[:2]) == {"high_risk_routing", "multiple_invoicing"} and abs(sum(scores.values()) - 1.0) < 1e-3
    assert suspected_typology(np.zeros(len(FEATURE_NAMES)), FEATURE_NAMES, none, severities)[0] == "undetermined"
    assert set(scores) == set(TYPOLOGIES)


# --------------------------------------------------------------------------- the alert pipeline end to end


@pytest.fixture(scope="module")
def pipeline(cfg, dataset, linked, tmp_path_factory):
    """Alerts for the small test dataset, with a quickly trained stand-in for the global model."""
    from tbml.data.generator import write_dataset

    write_dataset(dataset, cfg.seed_dir("data_dir", 0))
    scenario = Scenario(3, "dirichlet", 0.5)
    prepare_clients(cfg, 0, scenario)
    clients = [load_client(scenario_dir(cfg, 0, scenario), c) for c in range(3)]
    X = np.concatenate([c.part("train").X for c in clients])
    y = np.concatenate([c.part("train").y for c in clients])
    Xv = np.concatenate([c.part("val").X for c in clients])
    yv = np.concatenate([c.part("val").y for c in clients])
    model, _ = fit_mlp(cfg, X, y, Xv, yv, seed=0, epochs=8)
    config = copy.deepcopy(cfg)
    config.set_path("ledger.auto_start_node", False)
    ledger = open_ledger(config, backend="python", fresh=True, state_dir=tmp_path_factory.mktemp("alert-ledger"))
    info = {"run_id": "test-run", "round": 2, "global_model_sha256": "ab" * 32, "global_model_tx": "py:1"}
    out = run_alert_pipeline(config, 0, model, info, clients, linked, ledger)
    return out, ledger, config


def test_alerts_have_everything_an_investigator_needs(pipeline) -> None:
    out, _, cfg = pipeline
    alerts = out["alerts"]
    assert len(alerts) > 10
    assert [a["risk_score"] for a in alerts] == sorted((a["risk_score"] for a in alerts), reverse=True)
    for a in alerts:
        assert a["trade_id"].startswith("INV-") and a["alert_id"] == "ALERT-" + a["trade_id"].split("-")[-1]
        assert float(cfg.scoring.alert_min_score) <= a["risk_score"] <= 100 and a["risk_band"] in BANDS[1:]
        assert 1 <= len(a["top_features"]) <= int(cfg.scoring.top_k_features)
        for f in a["top_features"]:
            assert f["feature"] in FEATURE_NAMES and f["direction"] in {"raises risk", "lowers risk"} and f["text"]
        assert all(r["rule_id"] in RULE_IDS for r in a["fired_rules"])
        assert a["suspected_typology"] in [*TYPOLOGIES, "undetermined"]
        assert a["institution"].startswith("fl_client_") and "review" in a["disclaimer"]
        assert len(a["ledger"]["sha256"]) == 64 and a["ledger"]["tx_id"] and a["ledger"]["status"] == "registered"
        assert len(a["nonce"]) == 32
        # no ground truth inside an alert
        assert not ({"is_illicit", "typology", "label", "network_id", "truth"} & set(a))
    assert len({a["alert_id"] for a in alerts}) == len(alerts)
    assert len({a["nonce"] for a in alerts}) == len(alerts)


def test_alerts_verify_on_the_ledger_and_tampering_is_detected(pipeline) -> None:
    out, ledger, _ = pipeline
    alert = out["alerts"][0]
    check = verify_alert(ledger, alert)
    assert check["on_ledger"] and check["matches_alert_id"] and check["registered_by"] == alert["institution"] and check["doc_type"] == "alert"
    assert alert_digest(alert) == alert["ledger"]["sha256"], "the ledger receipt itself is not part of the hash"

    for change in [{"risk_score": alert["risk_score"] - 30}, {"risk_band": "Low"}, {"suspected_typology": "none"}, {"fired_rules": []}, {"trade_id": "INV-0000000"}]:
        forged = {**alert, **change}
        assert not verify_alert(ledger, forged)["on_ledger"], change
    downgraded = copy.deepcopy(alert)
    downgraded["top_features"][0]["text"] = "nothing unusual"
    assert not verify_alert(ledger, downgraded)["on_ledger"]
    # registering the same alert again changes nothing
    assert register_alerts(ledger, [copy.deepcopy(alert)])["alerts_registered"] == 1
    assert ledger.verify(alert_digest(alert)).ref_id == alert["alert_id"]


def test_scoring_evaluation(pipeline) -> None:
    out, _, cfg = pipeline
    ev = out["summary"]["evaluation"]
    assert [b["band"] for b in ev["bands"]] == BANDS
    assert sum(b["trades"] for b in ev["bands"]) == int((out["scores"]["split"] == "test").sum())
    low, critical = ev["bands"][0], ev["bands"][-1]
    assert low["precision"] < 0.15, "the Low band must be mostly legitimate trades"
    if critical["trades"] >= 10:
        assert critical["precision"] > 0.5
    assert set(ev["comparison"]) == {"risk_score", "model_only", "rules_only"}
    assert ev["comparison"]["risk_score"]["pr_auc"] > ev["comparison"]["rules_only"]["pr_auc"]
    assert ev["alerts"]["total"] == len(out["alerts"]) == sum(ev["alerts"]["by_band"].values())
    assert 0.0 <= ev["typology_attribution"]["top1_accuracy"] <= ev["typology_attribution"]["top2_accuracy"] <= 1.0
    assert len(ev["weight_sensitivity"]) == 11
    assert out["summary"]["shap"]["additivity_error"] < 0.05
    scores = out["scores"]
    assert set(scores["risk_band"]) <= set(BANDS) and scores["risk_score"].between(0, 100).all()
    assert scores["trade_id"].is_unique


def test_report_is_readable(pipeline) -> None:
    out, _, cfg = pipeline
    report = out["report"]
    assert report.startswith("# Alert report") and "not a finding of laundering" in report
    assert "| Critical |" in report and "Why the model flagged it" in report and "Rules fired:" in report
    assert report.count("### ALERT-") == min(int(cfg.scoring.report_sample), len(out["alerts"]))
    sample = sample_alerts(out["alerts"], 6)
    assert len(sample) == min(6, len(out["alerts"])) and len({a["alert_id"] for a in sample}) == len(sample)
    text = alert_markdown(out["alerts"][0], truth="over_invoicing")
    assert "Evidence: SHA-256" in text and "Ground truth, for evaluation only" in text
    assert "Ground truth" not in alert_markdown(out["alerts"][0])
