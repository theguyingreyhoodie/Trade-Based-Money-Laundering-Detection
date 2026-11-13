"""Feature dictionary: every risk indicator, what it measures and which typology it targets.

This table is the single source of truth for feature names and order. It also
drives ``docs/features.md`` and the plain-English wording of alert explanations.

``owner`` says which data owner could compute the feature alone
(``bank``, ``trade``, ``customs``) or whether it needs linked records
(``cross``). ``scope`` is ``row`` for indicators computed from one trade's own
documents and ``context`` for indicators that depend on the other trades the
computing party can see (entity history, trade graph).
"""

from __future__ import annotations

from dataclasses import dataclass

T_OVER = "over_invoicing"
T_UNDER = "under_invoicing"
T_MULTI = "multiple_invoicing"
T_PHANTOM = "phantom_shipment"
T_SHIP = "over_under_shipment"
T_MISDESC = "misdescription"
T_THIRD = "third_party_payment"
T_CIRC = "circular_trading"
T_SHELL = "shell_company"
T_ROUTE = "high_risk_routing"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str  # invoice | shipment | payment | counterparty | network
    scope: str  # row | context
    owner: str  # bank | trade | customs | cross
    description: str
    typologies: tuple[str, ...]


_F = FeatureSpec

FEATURES: list[FeatureSpec] = [
    # ------------------------------------------------------------------ invoice / price
    _F("price_log_ratio", "invoice", "row", "trade", "log of unit price over the HS reference price (signed)", (T_OVER, T_UNDER)),
    _F("price_z", "invoice", "row", "trade", "price_log_ratio divided by the HS reference dispersion (signed z-score)", (T_OVER, T_UNDER)),
    _F("price_abs_z", "invoice", "row", "trade", "absolute price z-score", (T_OVER, T_UNDER, T_MISDESC)),
    _F("price_abs_z_declared_hs", "invoice", "row", "cross", "absolute price z-score against the reference of the HS code declared to customs", (T_MISDESC,)),
    _F("price_dev_exporter_hs", "invoice", "context", "trade", "log of unit price over the exporter's own median price for this HS code", (T_OVER, T_UNDER)),
    _F("invoice_value_log", "invoice", "row", "trade", "log10 of the invoice value in USD", (T_SHELL, T_OVER)),
    _F("round_amount_level", "invoice", "row", "trade", "number of trailing zeros of a whole-number invoice total (0 if not whole)", (T_SHELL,)),
    _F("round_amount_flag", "invoice", "row", "trade", "invoice total is a multiple of the round-amount base", (T_SHELL,)),
    _F("inv_pay_gap", "invoice", "row", "cross", "(amount paid - invoice value) / invoice value", (T_MULTI, T_THIRD)),
    _F("inv_pay_abs_gap", "invoice", "row", "cross", "absolute relative gap between amount paid and invoice value", (T_MULTI, T_THIRD)),
    _F("customs_value_log_ratio", "invoice", "row", "cross", "log of customs declared value over invoice value", (T_OVER, T_UNDER, T_MISDESC)),
    _F("customs_value_abs_dev", "invoice", "row", "cross", "absolute size of the gap between customs value and invoice value", (T_OVER, T_UNDER, T_MISDESC)),
    # ------------------------------------------------------------------ shipment / customs
    _F("missing_shipment", "shipment", "row", "cross", "no shipment record could be linked to the invoice", (T_PHANTOM,)),
    _F("missing_customs", "shipment", "row", "cross", "no customs declaration could be linked to the invoice", (T_PHANTOM,)),
    _F("missing_both", "shipment", "row", "cross", "neither shipment nor customs record exists", (T_PHANTOM,)),
    _F("ship_qty_log_ratio", "shipment", "context", "cross", "log of shipped quantity over the quantity invoiced against the same bill of lading", (T_SHIP, T_MULTI)),
    _F("ship_qty_abs_dev", "shipment", "context", "cross", "absolute size of the gap between shipped and invoiced quantity", (T_SHIP, T_MULTI)),
    _F("customs_qty_log_ratio", "shipment", "row", "cross", "log of customs declared quantity over invoiced quantity", (T_SHIP,)),
    _F("customs_qty_abs_dev", "shipment", "row", "cross", "absolute size of the gap between declared and invoiced quantity", (T_SHIP,)),
    _F("weight_per_unit_log_ratio", "shipment", "row", "cross", "log of shipped weight per unit over the typical weight of the invoiced goods", (T_MISDESC, T_PHANTOM, T_SHIP)),
    _F("weight_per_unit_abs_dev", "shipment", "row", "cross", "absolute size of the shipped weight-per-unit anomaly", (T_MISDESC, T_PHANTOM, T_SHIP)),
    _F("customs_weight_per_unit_log_ratio", "shipment", "row", "cross", "log of declared weight per unit over the typical weight of the invoiced goods", (T_MISDESC,)),
    _F("hs_mismatch", "shipment", "row", "cross", "invoice HS code differs from the customs HS code", (T_MISDESC,)),
    _F("hs_heading_mismatch", "shipment", "row", "cross", "invoice and customs HS codes differ in the 4-digit heading", (T_MISDESC,)),
    _F("hs_chapter_mismatch", "shipment", "row", "cross", "invoice and customs HS codes differ in the 2-digit chapter", (T_MISDESC,)),
    _F("n_transshipments", "shipment", "row", "trade", "number of transshipment stops", (T_ROUTE,)),
    _F("route_detour_ratio", "shipment", "row", "trade", "route length over the direct distance", (T_ROUTE,)),
    _F("route_via_high_risk", "shipment", "row", "trade", "number of stops in listed high-risk jurisdictions", (T_ROUTE,)),
    _F("route_via_ftz", "shipment", "row", "trade", "number of stops in free-trade-zone / offshore jurisdictions", (T_ROUTE,)),
    _F("route_max_stop_risk", "shipment", "row", "trade", "highest jurisdiction risk score among the stops", (T_ROUTE,)),
    _F("route_rarity", "shipment", "context", "trade", "-log share of this corridor's trades that used the same route", (T_ROUTE,)),
    _F("days_invoice_to_ship", "shipment", "row", "trade", "days from invoice to departure", (T_CIRC, T_PHANTOM)),
    _F("invoices_per_bl", "shipment", "context", "cross", "number of invoices quoting the same bill of lading", (T_MULTI,)),
    _F("bl_payer_banks", "shipment", "context", "cross", "number of distinct paying banks across invoices on the same bill of lading", (T_MULTI,)),
    # ------------------------------------------------------------------ payment
    _F("third_party_payer", "payment", "row", "bank", "a payer is not the importer", (T_THIRD,)),
    _F("n_distinct_payers", "payment", "row", "bank", "number of distinct payers", (T_THIRD,)),
    _F("payer_country_differs", "payment", "row", "bank", "third-party payer is in a different jurisdiction than the importer", (T_THIRD,)),
    _F("payer_risk", "payment", "row", "bank", "highest jurisdiction risk score among third-party payers", (T_THIRD,)),
    _F("payer_related", "payment", "row", "bank", "third-party payer shares an address or director with the importer (benign signal)", (T_THIRD,)),
    _F("payer_is_finance_house", "payment", "row", "bank", "third-party payer is a registered trade-finance house (benign signal)", (T_THIRD,)),
    _F("payer_age_log", "payment", "row", "bank", "log1p age in days of the youngest third-party payer (0 if none)", (T_THIRD, T_SHELL)),
    _F("days_invoice_to_payment", "payment", "row", "cross", "days from invoice to first payment", (T_PHANTOM, T_SHELL, T_CIRC)),
    _F("days_payment_minus_ship", "payment", "row", "cross", "days from departure to first payment (negative = paid before shipment)", (T_PHANTOM, T_SHELL)),
    _F("paid_before_shipment", "payment", "row", "cross", "first payment precedes departure", (T_PHANTOM, T_SHELL)),
    _F("n_payments", "payment", "row", "bank", "number of payment instalments", (T_THIRD,)),
    _F("payment_span_days", "payment", "row", "bank", "days between first and last instalment", (T_THIRD,)),
    _F("currency_mismatch", "payment", "row", "cross", "payment currency differs from invoice currency", (T_THIRD,)),
    _F("currency_unusual", "payment", "row", "trade", "invoice currency is neither USD nor a home currency of the parties", (T_THIRD, T_ROUTE)),
    _F("payee_bank_offshore", "payment", "row", "bank", "exporter is paid into a bank outside its own jurisdiction", (T_ROUTE, T_SHELL)),
    _F("payee_bank_risk", "payment", "row", "bank", "jurisdiction risk score of that bank (0 if domestic)", (T_ROUTE, T_SHELL)),
    _F("payer_bank_offshore", "payment", "row", "bank", "payer pays from a bank outside its own jurisdiction", (T_MULTI, T_THIRD)),
    # ------------------------------------------------------------------ counterparty
    _F("exporter_age_log", "counterparty", "row", "bank", "log1p age in days of the exporter at invoice date", (T_SHELL,)),
    _F("importer_age_log", "counterparty", "row", "bank", "log1p age in days of the importer at invoice date", (T_SHELL,)),
    _F("min_party_age_log", "counterparty", "row", "bank", "log1p age in days of the younger party", (T_SHELL,)),
    _F("exporter_country_risk", "counterparty", "row", "bank", "jurisdiction risk score of the exporter", (T_ROUTE, T_SHELL)),
    _F("importer_country_risk", "counterparty", "row", "bank", "jurisdiction risk score of the importer", (T_ROUTE, T_SHELL)),
    _F("max_party_risk", "counterparty", "row", "bank", "higher jurisdiction risk score of the two parties", (T_ROUTE, T_SHELL)),
    _F("parties_related", "counterparty", "row", "bank", "exporter and importer share an address or a director", (T_CIRC, T_SHELL)),
    _F("max_addr_shared_log", "counterparty", "context", "bank", "log1p number of other known entities at a party's address (max of both parties)", (T_SHELL,)),
    _F("max_dir_shared_log", "counterparty", "context", "bank", "log1p number of other known entities with a party's director (max of both parties)", (T_SHELL, T_CIRC)),
    _F("exporter_n_trades_log", "counterparty", "context", "trade", "log1p number of trades of the exporter seen so far", (T_SHELL,)),
    _F("importer_n_trades_log", "counterparty", "context", "trade", "log1p number of trades of the importer seen so far", (T_SHELL,)),
    _F("max_volume_per_age", "counterparty", "context", "cross", "log10 of traded USD volume per year of age (max of both parties)", (T_SHELL,)),
    _F("pair_n_trades_log", "counterparty", "context", "trade", "log1p number of trades between this exporter and importer", (T_CIRC, T_MULTI)),
    _F("pair_share_exporter", "counterparty", "context", "trade", "share of the exporter's volume that goes to this importer", (T_CIRC, T_SHELL)),
    _F("exporter_hhi", "counterparty", "context", "trade", "concentration (HHI) of the exporter's volume across importers", (T_CIRC, T_SHELL)),
    _F("importer_hhi", "counterparty", "context", "trade", "concentration (HHI) of the importer's volume across exporters", (T_CIRC, T_SHELL)),
    _F("max_hs_diversity", "counterparty", "context", "trade", "number of distinct HS chapters traded by a party (max of both parties)", (T_SHELL,)),
    _F("exporter_import_share", "counterparty", "context", "trade", "share of the exporter's trades in which it is the buyer", (T_CIRC,)),
    _F("importer_export_share", "counterparty", "context", "trade", "share of the importer's trades in which it is the seller", (T_CIRC,)),
    # ------------------------------------------------------------------ network (graph)
    _F("exporter_out_degree_log", "network", "context", "trade", "log1p number of distinct buyers of the exporter", (T_SHELL, T_CIRC)),
    _F("exporter_in_degree_log", "network", "context", "trade", "log1p number of distinct suppliers of the exporter", (T_CIRC,)),
    _F("importer_in_degree_log", "network", "context", "trade", "log1p number of distinct suppliers of the importer", (T_SHELL, T_CIRC)),
    _F("importer_out_degree_log", "network", "context", "trade", "log1p number of distinct buyers of the importer", (T_CIRC,)),
    _F("exporter_pagerank_log", "network", "context", "trade", "log of the exporter's PageRank in the trade graph (scaled by graph size)", (T_CIRC, T_SHELL)),
    _F("importer_pagerank_log", "network", "context", "trade", "log of the importer's PageRank in the trade graph (scaled by graph size)", (T_CIRC, T_SHELL)),
    _F("edge_reciprocal", "network", "context", "trade", "the importer also sells to the exporter", (T_CIRC,)),
    _F("edge_in_cycle", "network", "context", "trade", "the trade lies on a directed cycle of the trade graph", (T_CIRC,)),
    _F("edge_in_cycle_same_hs", "network", "context", "trade", "the trade lies on a directed cycle formed by trades in the same HS code", (T_CIRC,)),
    _F("cycle_len_same_hs", "network", "context", "trade", "length of the shortest same-HS cycle through the trade (0 if none)", (T_CIRC,)),
    _F("exporter_resale_days_log", "network", "context", "trade", "log1p days since the exporter last bought the same HS code (365 if never): rapid resale", (T_CIRC, T_SHELL)),
]

FEATURE_NAMES: list[str] = [f.name for f in FEATURES]
FEATURE_INDEX: dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}
FEATURE_BY_NAME: dict[str, FeatureSpec] = {f.name: f for f in FEATURES}
GROUPS = ["invoice", "shipment", "payment", "counterparty", "network"]
ROW_FEATURES = [f.name for f in FEATURES if f.scope == "row"]
CONTEXT_FEATURES = [f.name for f in FEATURES if f.scope == "context"]


def features_markdown() -> str:
    """Render the feature dictionary as Markdown (written to ``docs/features.md``)."""
    lines = [
        "# Feature dictionary",
        "",
        "Generated from `src/tbml/features/dictionary.py`; do not edit by hand (run `python scripts/build_features.py --docs-only`).",
        "",
        f"{len(FEATURES)} risk indicators in five groups. **Scope** is `row` when the indicator uses only the trade's own",
        "documents and `context` when it depends on the other trades visible to whoever computes it (entity history, trade graph).",
        "Context indicators are computed per FL client from that client's own trades only. **Owner** names the data owner that",
        "could compute the indicator alone; `cross` needs records linked across owners.",
        "",
        "Design notes:",
        "",
        "- No identifiers (entity, bank, HS code, country names) are used as features; only behaviour and public risk tiers.",
        "- Ratios are log-ratios clipped to a fixed range; counts are `log1p`-transformed. Missing documents give neutral",
        "  values plus an explicit `missing_*` flag, so that absence is a signal and not an imputation artefact.",
        "- Context indicators for a trade use only trades up to the end of that trade's split period (train, validation or",
        "  test), and never use labels.",
        "",
    ]
    titles = {"invoice": "Invoice / price", "shipment": "Shipment / customs", "payment": "Payment", "counterparty": "Counterparty", "network": "Trade network (graph)"}
    for group in GROUPS:
        lines += [f"## {titles[group]}", "", "| Feature | Scope | Owner | What it measures | Typologies targeted |", "|---|---|---|---|---|"]
        for f in FEATURES:
            if f.group == group:
                lines.append(f"| `{f.name}` | {f.scope} | {f.owner} | {f.description} | {', '.join(t.replace('_', ' ') for t in f.typologies)} |")
        lines.append("")
    lines += ["## Typology coverage", "", "| Typology | Indicators |", "|---|---|"]
    from tbml.data.typologies import TYPOLOGIES

    for typology in TYPOLOGIES:
        names = [f"`{f.name}`" for f in FEATURES if typology in f.typologies]
        lines.append(f"| {typology.replace('_', ' ')} | {', '.join(names)} |")
    lines.append("")
    return "\n".join(lines)
