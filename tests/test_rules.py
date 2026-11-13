"""Rules engine: each rule fires on its red flag, thresholds come from the config."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from tbml.data.typologies import TYPOLOGIES
from tbml.features.dictionary import FEATURE_INDEX, FEATURE_NAMES
from tbml.models.baselines import rules_baseline
from tbml.rules.engine import RULE_BY_ID, RULE_IDS, RULES, fired_rules, rule_hits, rule_score, rule_severities, rules_markdown


def neutral_row() -> np.ndarray:
    """An unremarkable trade: old parties, direct route, price at reference, everything matched."""
    x = np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    def put(name: str, value: float) -> None:
        x[FEATURE_INDEX[name]] = value

    put("min_party_age_log", np.log1p(4000))
    put("exporter_age_log", np.log1p(4000))
    put("importer_age_log", np.log1p(5000))
    put("payer_age_log", np.log1p(5000))
    put("route_detour_ratio", 1.0)
    put("invoices_per_bl", 1.0)
    put("bl_payer_banks", 1.0)
    put("n_payments", 1.0)
    put("n_distinct_payers", 1.0)
    put("days_payment_minus_ship", 20.0)
    put("days_invoice_to_payment", 30.0)
    put("exporter_resale_days_log", np.log1p(365))
    put("younger_party_volume_per_age", 6.0)
    put("max_party_risk", 0.1)
    return x


# feature values that should trigger exactly the named rule on top of a neutral trade
TRIGGERS: dict[str, dict[str, float]] = {
    "R01": {"price_log_ratio": np.log(3.0), "price_z": 4.0},
    "R02": {"price_log_ratio": np.log(0.3), "price_z": -4.0},
    "R03": {"customs_value_abs_dev": 0.5},
    "R04": {"invoices_per_bl": 2.0, "ship_qty_log_ratio": -0.69},
    "R05": {"missing_both": 1.0, "missing_customs": 1.0, "missing_shipment": 1.0},
    "R06": {"missing_customs": 1.0, "days_payment_minus_ship": -12.0},
    "R07": {"customs_qty_abs_dev": 0.6},
    "R08": {"hs_heading_mismatch": 1.0},
    "R09": {"weight_per_unit_abs_dev": 1.5},
    "R10": {"third_party_payer": 1.0, "payer_risk": 1.0},
    "R11": {"third_party_payer": 1.0, "payer_country_differs": 1.0, "payer_risk": 0.1},
    "R12": {"edge_in_cycle_same_hs": 1.0, "cycle_len_same_hs": 3.0},
    "R13": {"edge_in_cycle_same_hs": 1.0, "cycle_len_same_hs": 2.0, "exporter_resale_days_log": np.log1p(5)},
    "R14": {"min_party_age_log": np.log1p(200), "max_addr_shared_log": np.log1p(8)},
    "R15": {"min_party_age_log": np.log1p(200), "younger_party_volume_per_age": 8.5},
    "R16": {"round_amount_flag": 1.0, "paid_before_shipment": 1.0, "min_party_age_log": np.log1p(500)},
    "R17": {"route_detour_ratio": 3.0},
    "R18": {"route_via_high_risk": 1.0},
    "R19": {"payee_bank_offshore": 1.0, "payee_bank_risk": 0.7},
    "R20": {"max_party_risk": 1.0},
}


def test_rule_book_is_complete() -> None:
    assert len(RULE_IDS) == len(set(RULE_IDS)) == len(RULES)
    assert set(TRIGGERS) == set(RULE_IDS), "every rule needs a trigger case in this test"
    covered = {r.typology for r in RULES}
    assert covered == set(TYPOLOGIES), f"typologies without a rule: {set(TYPOLOGIES) - covered}"
    for rule in RULES:
        assert 0.0 < rule.severity < 1.0
        assert rule.description and rule.name


def test_neutral_trade_fires_nothing(cfg) -> None:
    hits = rule_hits(neutral_row()[None, :], cfg)
    assert hits.shape == (1, len(RULES))
    assert not hits.any(), fired_rules(hits[0])
    assert rule_score(hits, cfg)[0] == 0.0


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_each_rule_fires_on_its_red_flag(cfg, rule_id: str) -> None:
    x = neutral_row()
    for name, value in TRIGGERS[rule_id].items():
        x[FEATURE_INDEX[name]] = value
    fired = fired_rules(rule_hits(x[None, :], cfg)[0])
    assert rule_id in fired, (rule_id, fired)
    assert rule_score(rule_hits(x[None, :], cfg), cfg)[0] >= RULE_BY_ID[rule_id].severity - 1e-9


def test_thresholds_come_from_config(cfg) -> None:
    x = neutral_row()
    x[FEATURE_INDEX["route_detour_ratio"]] = 1.5
    j = RULE_IDS.index("R17")
    assert not rule_hits(x[None, :], cfg)[0, j]
    strict = copy.deepcopy(cfg)
    strict.set_path("rules.thresholds.detour_ratio_high", 1.2)
    assert rule_hits(x[None, :], strict)[0, j]


def test_score_is_noisy_or(cfg) -> None:
    severity = rule_severities(cfg)
    hits = np.zeros((3, len(RULES)), dtype=bool)
    hits[1, [0, 4]] = True
    hits[2, :] = True
    score = rule_score(hits, cfg)
    assert score[0] == 0.0
    assert abs(score[1] - (1 - (1 - severity[0]) * (1 - severity[4]))) < 1e-12
    assert score[1] > max(severity[0], severity[4])
    assert score[1] < score[2] < 1.0

    custom = copy.deepcopy(cfg)
    custom.set_path("rules.severity", {"R01": 0.99})
    assert rule_severities(custom)[0] == 0.99
    assert rule_score(hits, custom)[1] > score[1]


def test_rules_baseline_on_generated_data(cfg, features) -> None:
    result, hits, score = rules_baseline(cfg, features)
    assert hits.shape == (len(features), len(RULES)) and hits.dtype == bool
    assert ((score >= 0) & (score < 1)).all()
    assert result["pr_auc"] > 3 * result["prevalence"], "rules must rank far better than chance"
    assert result["recall"] > 0.3
    assert len(result["per_rule"]) == len(RULES)
    # rules catch a much larger share of illicit than of legitimate trades
    any_hit = hits.any(axis=1)
    assert any_hit[features.y == 1].mean() > 2 * any_hit[features.y == 0].mean()
    assert "| R01 |" in rules_markdown(cfg)
