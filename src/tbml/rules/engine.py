"""Transparent red-flag rules engine (Step 03, objective 2).

Each rule is a readable predicate over the risk indicators, with thresholds
taken from ``rules.thresholds`` in the config. The engine reports which rules
fired for every trade and combines them into one score.

Rule severities are expert judgements (how strongly a hit on its own suggests
laundering), not parameters fitted to labels. The rule score is a noisy-OR of
the severities of the fired rules: ``1 - prod(1 - severity)``. One strong rule
is enough for a high score, and several weak rules add up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from tbml.config import Config
from tbml.features.dictionary import FEATURE_INDEX


class FeatureView:
    """Column access by feature name over a feature matrix."""

    def __init__(self, X: np.ndarray, names: list[str] | None = None) -> None:
        self.X = X
        self.index = FEATURE_INDEX if names is None else {name: i for i, name in enumerate(names)}

    def __getitem__(self, name: str) -> np.ndarray:
        return self.X[:, self.index[name]]


@dataclass(frozen=True)
class Rule:
    rule_id: str
    name: str
    typology: str
    severity: float
    description: str
    predicate: Callable[[FeatureView, Config], np.ndarray]


def _days(log_days: np.ndarray) -> np.ndarray:
    return np.expm1(log_days)


def _unrelated_third_party(f: FeatureView) -> np.ndarray:
    return (f["third_party_payer"] > 0) & (f["payer_related"] == 0) & (f["payer_is_finance_house"] == 0)


RULES: list[Rule] = [
    Rule("R01", "Unit price far above reference", "over_invoicing", 0.60,
         "Unit price exceeds the HS reference by more than the price-ratio and z-score thresholds.",
         lambda f, t: (f["price_log_ratio"] > np.log(t.price_ratio_high)) & (f["price_z"] > t.price_z_high)),
    Rule("R02", "Unit price far below reference", "under_invoicing", 0.60,
         "Unit price is below the HS reference by more than the price-ratio and z-score thresholds.",
         lambda f, t: (f["price_log_ratio"] < np.log(t.price_ratio_low)) & (f["price_z"] < -t.price_z_high)),
    Rule("R03", "Customs value differs from invoice value", "over_invoicing", 0.50,
         "The value declared to customs and the invoiced value differ by more than the tolerance.",
         lambda f, t: f["customs_value_abs_dev"] > t.value_mismatch_customs),
    Rule("R04", "Several invoices exceed one shipment", "multiple_invoicing", 0.90,
         "Two or more invoices quote the same bill of lading and together exceed the shipped quantity.",
         lambda f, t: (f["invoices_per_bl"] >= t.invoices_per_bl_min) & (f["ship_qty_log_ratio"] < -np.log(t.bl_cover_ratio_high))),
    Rule("R05", "Paid, but no shipment and no customs record", "phantom_shipment", 0.70,
         "Payment was made although neither a shipment nor a customs declaration can be found.",
         lambda f, t: f["missing_both"] > 0),
    Rule("R06", "No customs record and paid early or near-empty shipment", "phantom_shipment", 0.50,
         "The customs declaration is missing and the trade was paid before shipment, or the cargo weighs far less than the goods should.",
         lambda f, t: (f["missing_customs"] > 0) & (f["missing_both"] == 0)
         & ((f["days_payment_minus_ship"] < t.advance_payment_days) | (f["weight_per_unit_log_ratio"] < -t.weight_per_unit_dev))),
    Rule("R07", "Shipped or declared quantity differs from invoice", "over_under_shipment", 0.60,
         "Shipped or customs-declared quantity deviates from the invoiced quantity by more than the tolerance.",
         lambda f, t: (f["ship_qty_abs_dev"] > t.qty_mismatch) | (f["customs_qty_abs_dev"] > t.qty_mismatch)),
    Rule("R08", "Goods declared under a different HS heading", "misdescription", 0.60,
         "The HS code declared to customs is in a different 4-digit heading than the invoiced goods.",
         lambda f, t: f["hs_heading_mismatch"] > 0),
    Rule("R09", "Cargo weight inconsistent with the goods", "misdescription", 0.40,
         "Weight per unit (shipped or declared) is far from the typical weight of the invoiced goods.",
         lambda f, t: (f["weight_per_unit_abs_dev"] > t.weight_per_unit_dev) | (np.abs(f["customs_weight_per_unit_log_ratio"]) > t.weight_per_unit_dev)),
    Rule("R10", "Unrelated third-party payer in a high-risk jurisdiction", "third_party_payment", 0.80,
         "The invoice was paid by a party that is neither related to the importer nor a finance house and sits in a high-risk or offshore jurisdiction.",
         lambda f, t: _unrelated_third_party(f) & (f["payer_risk"] >= t.third_party_risk_min)),
    Rule("R11", "Unrelated third-party payer abroad", "third_party_payment", 0.40,
         "The invoice was paid by an unrelated party in a different jurisdiction than the importer.",
         lambda f, t: _unrelated_third_party(f) & (f["payer_country_differs"] > 0) & (f["payer_risk"] < t.third_party_risk_min)),
    Rule("R12", "Same goods traded round a loop", "circular_trading", 0.70,
         "The trade lies on a cycle of three or more parties trading the same HS code.",
         lambda f, t: (f["edge_in_cycle_same_hs"] > 0) & (f["cycle_len_same_hs"] >= 3)),
    Rule("R13", "Goods resold within days inside a loop", "circular_trading", 0.40,
         "The seller bought the same goods only days earlier and the trade lies on a cycle of trades in that HS code.",
         lambda f, t: (f["edge_in_cycle_same_hs"] > 0) & (_days(f["exporter_resale_days_log"]) <= t.rapid_resale_days)),
    Rule("R14", "Young company at a mass-registration address", "shell_company", 0.60,
         "A party is younger than the threshold and shares its address with several other entities.",
         lambda f, t: (_days(f["min_party_age_log"]) < t.young_entity_days) & (np.expm1(f["max_addr_shared_log"]) >= t.shared_address_min)),
    Rule("R15", "Young company with very high turnover", "shell_company", 0.50,
         "A party is younger than the threshold but trades an unusually high volume for its age.",
         lambda f, t: (_days(f["min_party_age_log"]) < t.young_entity_days) & (f["younger_party_volume_per_age"] > t.volume_per_age_high)),
    Rule("R16", "Round amount paid in advance to or by a young company", "shell_company", 0.40,
         "A round invoice total, payment before shipment and a party younger than twice the age threshold.",
         lambda f, t: (f["round_amount_flag"] > 0) & (f["paid_before_shipment"] > 0) & (_days(f["min_party_age_log"]) < 2 * t.young_entity_days)),
    Rule("R17", "Needless transshipment", "high_risk_routing", 0.50,
         "The route has many stops or is much longer than the direct route.",
         lambda f, t: (f["n_transshipments"] >= t.transshipments_high) | (f["route_detour_ratio"] > t.detour_ratio_high)),
    Rule("R18", "Stop in a listed high-risk jurisdiction", "high_risk_routing", 0.50,
         "The goods were routed through a jurisdiction on the high-risk list.",
         lambda f, t: f["route_via_high_risk"] > 0),
    Rule("R19", "Proceeds collected in an offshore bank", "high_risk_routing", 0.40,
         "The exporter is paid into a bank outside its own jurisdiction, in a high-risk or offshore centre.",
         lambda f, t: (f["payee_bank_offshore"] > 0) & (f["payee_bank_risk"] >= t.third_party_risk_min)),
    Rule("R20", "Party in a listed high-risk jurisdiction", "high_risk_routing", 0.10,
         "The exporter or the importer is established in a jurisdiction on the high-risk list.",
         lambda f, t: f["max_party_risk"] >= 1.0),
]

RULE_IDS: list[str] = [r.rule_id for r in RULES]
RULE_BY_ID: dict[str, Rule] = {r.rule_id: r for r in RULES}


def rule_severities(cfg: Config) -> np.ndarray:
    """Severity per rule; ``rules.severity`` in the config overrides the defaults."""
    overrides = cfg.rules.get("severity", {}) or {}
    return np.array([float(overrides.get(r.rule_id, r.severity)) for r in RULES])


def rule_hits(X: np.ndarray, cfg: Config, names: list[str] | None = None) -> np.ndarray:
    """Boolean matrix [n_trades, n_rules]: which rules fired for which trade."""
    view = FeatureView(X, names)
    thresholds = cfg.rules.thresholds
    return np.column_stack([np.asarray(rule.predicate(view, thresholds), dtype=bool) for rule in RULES])


def rule_score(hits: np.ndarray, cfg: Config) -> np.ndarray:
    """Noisy-OR combination of the severities of the fired rules, in [0, 1)."""
    severity = rule_severities(cfg)
    return 1.0 - np.prod(np.where(hits, 1.0 - severity[None, :], 1.0), axis=1)


def fired_rules(hits_row: np.ndarray) -> list[str]:
    """Ids of the rules that fired for one trade."""
    return [RULE_IDS[j] for j in np.flatnonzero(hits_row)]


def rules_markdown(cfg: Config) -> str:
    """Rule book as a Markdown table (used in the docs and reports)."""
    severity = rule_severities(cfg)
    lines = ["| Rule | Red flag | Typology | Severity | Description |", "|---|---|---|---|---|"]
    for rule, sev in zip(RULES, severity):
        lines.append(f"| {rule.rule_id} | {rule.name} | {rule.typology.replace('_', ' ')} | {sev:.2f} | {rule.description} |")
    return "\n".join(lines)


def rule_book_markdown(cfg: Config) -> str:
    """The full rule book document (``docs/rules.md``)."""
    lines = [
        "# Rules engine: red-flag rule book",
        "",
        "Generated from `src/tbml/rules/engine.py` and `configs/default.yaml` (run `python scripts/run_baselines.py --docs-only`).",
        "",
        f"{len(RULES)} transparent rules over the risk indicators in `docs/features.md`. A rule either fires for a trade or",
        "it does not; the engine reports the fired rules per trade, so every alert can be read without a model.",
        "",
        "- **Thresholds** live in `rules.thresholds` in the config. They are expert settings informed by the unlabelled",
        "  distribution of each indicator (for example, a price z-score beyond 3 occurs in well under 1% of trades).",
        "  They are not fitted to labels.",
        "- **Severity** is an expert judgement of how strongly one hit suggests laundering (0 to 1). It can be overridden",
        "  per rule with `rules.severity`.",
        "- **Rule score** = `1 - prod(1 - severity)` over the fired rules (noisy-OR). A single strong rule gives a high",
        "  score and several weak rules add up, but the score never reaches 1.",
        "- As a stand-alone baseline, a trade is flagged when its rule score reaches a threshold chosen on the",
        "  validation period. Measured quality per rule is in `results/RESULTS.md`.",
        "",
        "## Rules",
        "",
        rules_markdown(cfg),
        "",
        "## Thresholds",
        "",
        "| Parameter | Value |",
        "|---|---|",
    ]
    lines += [f"| `{key}` | {value} |" for key, value in cfg.rules.thresholds.items()]
    lines.append("")
    return "\n".join(lines)
