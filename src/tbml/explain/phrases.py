"""Plain-English wording of risk indicators for investigators.

``describe(name, value)`` turns one indicator value into a short phrase such as
"unit price 3.4x the HS reference price". Every feature in the dictionary has a
phrase (checked by a test), so an alert never shows a raw column name.
"""

from __future__ import annotations

import math
from typing import Callable

from tbml.features.dictionary import FEATURE_BY_NAME


def _times(log_ratio: float) -> str:
    ratio = math.exp(log_ratio)
    if 0.95 < ratio < 1.05:
        return "about equal to"
    if ratio < 0.095:
        return f"{ratio:.2g}x"
    return f"{ratio:.0f}x" if ratio >= 20 else f"{ratio:.1f}x"


def _age(log_days: float) -> str:
    days = math.expm1(log_days)
    if days < 60:
        return f"{days:.0f} days"
    if days < 730:
        return f"{days / 30.4:.0f} months"
    return f"{days / 365.25:.0f} years"


def _count(log_count: float) -> int:
    return int(round(math.expm1(log_count)))


def _plural(n: float, word: str) -> str:
    n = int(round(n))
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _entities(n: int) -> str:
    return "1 other known entity" if n == 1 else f"{n} other known entities"


def _flag(yes: str, no: str) -> Callable[[float], str]:
    return lambda v: yes if v > 0.5 else no


def _signed_days(v: float, after: str, before: str, same: str) -> str:
    if abs(v) < 0.5:
        return same
    days = int(round(abs(v)))
    return f"{days} {'day' if days == 1 else 'days'} {after if v > 0 else before}"


PHRASES: dict[str, Callable[[float], str]] = {
    # ---- invoice / price
    "price_log_ratio": lambda v: f"unit price {_times(v)} the HS reference price",
    "price_z": lambda v: f"unit price {abs(v):.1f} standard deviations {'above' if v >= 0 else 'below'} the HS reference",
    "price_abs_z": lambda v: f"unit price {v:.1f} standard deviations away from the HS reference",
    "price_abs_z_declared_hs": lambda v: f"unit price {v:.1f} standard deviations away from the reference of the goods declared to customs",
    "price_dev_exporter_hs": lambda v: f"unit price {_times(v)} this exporter's usual price for the goods",
    "invoice_value_log": lambda v: f"invoice value about USD {10 ** v:,.0f}",
    "round_amount_level": lambda v: f"invoice total ends in {int(v)} zeros" if v >= 1 else "invoice total is not a round amount",
    "round_amount_flag": _flag("invoice total is a round amount", "invoice total is not a round amount"),
    "inv_pay_gap": lambda v: f"amount paid is {abs(v):.0%} {'more' if v > 0 else 'less'} than invoiced" if abs(v) >= 0.005 else "amount paid equals the invoice",
    "inv_pay_abs_gap": lambda v: f"amount paid differs from the invoice by {v:.0%}" if v >= 0.005 else "amount paid equals the invoice",
    "customs_value_log_ratio": lambda v: f"value declared to customs is {_times(v)} the invoice value",
    "customs_value_abs_dev": lambda v: f"customs value and invoice value differ by a factor of {math.exp(v):.2f}" if v >= 0.05 else "customs value agrees with the invoice",
    # ---- shipment / customs
    "missing_shipment": _flag("no shipment record can be linked to the invoice", "a shipment record exists"),
    "missing_customs": _flag("no customs declaration can be linked to the invoice", "a customs declaration exists"),
    "missing_both": _flag("neither a shipment record nor a customs declaration exists", "shipment and customs records are not both missing"),
    "ship_qty_log_ratio": lambda v: f"shipped quantity is {_times(v)} the quantity invoiced against this bill of lading",
    "ship_qty_abs_dev": lambda v: f"shipped and invoiced quantity differ by a factor of {math.exp(v):.2f}" if v >= 0.05 else "shipped quantity matches the invoice",
    "customs_qty_log_ratio": lambda v: f"quantity declared to customs is {_times(v)} the invoiced quantity",
    "customs_qty_abs_dev": lambda v: f"declared and invoiced quantity differ by a factor of {math.exp(v):.2f}" if v >= 0.05 else "declared quantity matches the invoice",
    "weight_per_unit_log_ratio": lambda v: f"cargo weight per unit is {_times(v)} the typical weight of the invoiced goods",
    "weight_per_unit_abs_dev": lambda v: f"cargo weight per unit is off by a factor of {math.exp(v):.2f}" if v >= 0.05 else "cargo weight fits the invoiced goods",
    "customs_weight_per_unit_log_ratio": lambda v: f"declared weight per unit is {_times(v)} the typical weight of the invoiced goods",
    "hs_mismatch": _flag("goods were declared to customs under a different HS code than invoiced", "customs HS code matches the invoice"),
    "hs_heading_mismatch": _flag("customs HS code is in a different 4-digit heading than the invoice", "customs HS heading matches the invoice"),
    "hs_chapter_mismatch": _flag("customs HS code is in a different chapter than the invoice", "customs HS chapter matches the invoice"),
    "n_transshipments": lambda v: f"{_plural(v, 'transshipment stop')}" if v >= 1 else "direct shipment",
    "route_detour_ratio": lambda v: f"route is {v:.1f}x the direct distance" if v >= 1.05 else "route is direct",
    "route_via_high_risk": lambda v: f"route passes through {_plural(v, 'listed high-risk jurisdiction')}" if v >= 1 else "no stop in a listed high-risk jurisdiction",
    "route_via_ftz": lambda v: f"route passes through {_plural(v, 'free-trade zone')}" if v >= 1 else "no stop in a free-trade zone",
    "route_max_stop_risk": lambda v: f"riskiest stop on the route has jurisdiction risk {v:.2f}" if v > 0 else "no stops on the route",
    "route_rarity": lambda v: "route is rarely used on this corridor" if v >= 2.5 else ("route is unusual on this corridor" if v >= 1.2 else "route is common on this corridor"),
    "days_invoice_to_ship": lambda v: f"shipped {_signed_days(v, 'after the invoice', 'before the invoice was issued', 'on the invoice date')}",
    "invoices_per_bl": lambda v: f"{int(v)} invoices quote the same bill of lading" if v >= 2 else "one invoice on the bill of lading",
    "bl_payer_banks": lambda v: f"invoices on this bill of lading were paid through {int(v)} different banks" if v >= 2 else "one paying bank for this bill of lading",
    # ---- payment
    "third_party_payer": _flag("paid by a third party, not the importer", "paid by the importer"),
    "n_distinct_payers": lambda v: f"{int(v)} different payers" if v >= 2 else "a single payer",
    "payer_country_differs": _flag("third-party payer is in a different jurisdiction than the importer", "no payer from another jurisdiction"),
    "payer_risk": lambda v: f"third-party payer in a jurisdiction with risk score {v:.2f}" if v > 0 else "no third-party payer",
    "payer_related": _flag("third-party payer shares an address or director with the importer", "payer is not a related party"),
    "payer_is_finance_house": _flag("paid by a registered trade-finance house", "payer is not a trade-finance house"),
    "payer_age_log": lambda v: f"youngest payer was registered {_age(v)} before the payment",
    "days_invoice_to_payment": lambda v: f"first payment {_signed_days(v, 'after the invoice', 'before the invoice', 'on the invoice date')}",
    "days_payment_minus_ship": lambda v: f"first payment {_signed_days(v, 'after departure', 'before departure', 'on the day of departure')}",
    "paid_before_shipment": _flag("paid before the goods were shipped", "paid after the goods were shipped"),
    "n_payments": lambda v: f"paid in {_plural(v, 'instalment')}",
    "payment_span_days": lambda v: f"instalments spread over {v:.0f} days" if v >= 1 else "paid in one go",
    "currency_mismatch": _flag("paid in a different currency than invoiced", "paid in the invoice currency"),
    "currency_unusual": _flag("invoice currency is neither USD nor a home currency of the parties", "usual invoice currency"),
    "payee_bank_offshore": _flag("exporter is paid into a bank outside its own jurisdiction", "exporter is paid into a domestic bank"),
    "payee_bank_risk": lambda v: f"exporter's collecting bank is in a jurisdiction with risk score {v:.2f}" if v > 0 else "exporter's collecting bank is domestic",
    "payer_bank_offshore": _flag("payer pays from a bank outside its own jurisdiction", "payer pays from a domestic bank"),
    # ---- counterparty
    "exporter_age_log": lambda v: f"exporter registered {_age(v)} before the invoice",
    "importer_age_log": lambda v: f"importer registered {_age(v)} before the invoice",
    "min_party_age_log": lambda v: f"younger party registered only {_age(v)} before the invoice" if math.expm1(v) < 730 else f"both parties are at least {_age(v)} old",
    "exporter_country_risk": lambda v: f"exporter's jurisdiction has risk score {v:.2f}",
    "importer_country_risk": lambda v: f"importer's jurisdiction has risk score {v:.2f}",
    "max_party_risk": lambda v: f"a party is in a jurisdiction with risk score {v:.2f}",
    "parties_related": _flag("exporter and importer share an address or a director", "exporter and importer are not related"),
    "max_addr_shared_log": lambda v: f"a party shares its address with {_entities(_count(v))}" if _count(v) >= 1 else "no party shares its address",
    "max_dir_shared_log": lambda v: f"a party shares a director with {_entities(_count(v))}" if _count(v) >= 1 else "no party shares a director",
    "exporter_n_trades_log": lambda v: f"exporter has {_plural(_count(v), 'trade')} on record",
    "importer_n_trades_log": lambda v: f"importer has {_plural(_count(v), 'trade')} on record",
    "younger_party_volume_per_age": lambda v: f"younger party trades about USD {10 ** v:,.0f} per year of existence",
    "pair_n_trades_log": lambda v: f"{_plural(_count(v), 'trade')} on record between these two parties",
    "pair_share_exporter": lambda v: f"{v:.0%} of the exporter's sales go to this importer",
    "exporter_hhi": lambda v: "exporter sells to very few buyers" if v >= 0.6 else ("exporter's sales are fairly concentrated" if v >= 0.3 else "exporter sells to many buyers"),
    "importer_hhi": lambda v: "importer buys from very few suppliers" if v >= 0.6 else ("importer's purchases are fairly concentrated" if v >= 0.3 else "importer buys from many suppliers"),
    "max_hs_diversity": lambda v: f"a party trades goods from {int(v)} different HS chapters",
    "exporter_import_share": lambda v: f"the seller is itself the buyer in {v:.0%} of its trades",
    "importer_export_share": lambda v: f"the buyer is itself the seller in {v:.0%} of its trades",
    # ---- trade network
    "exporter_out_degree_log": lambda v: f"exporter sells to {_plural(_count(v), 'buyer')}",
    "exporter_in_degree_log": lambda v: f"exporter buys from {_plural(_count(v), 'supplier')}",
    "importer_in_degree_log": lambda v: f"importer buys from {_plural(_count(v), 'supplier')}",
    "importer_out_degree_log": lambda v: f"importer sells to {_plural(_count(v), 'buyer')}",
    "exporter_pagerank_log": lambda v: "exporter is a hub of the trade network" if v >= 1.0 else ("exporter is peripheral in the trade network" if v <= -1.0 else "exporter has an ordinary position in the trade network"),
    "importer_pagerank_log": lambda v: "importer is a hub of the trade network" if v >= 1.0 else ("importer is peripheral in the trade network" if v <= -1.0 else "importer has an ordinary position in the trade network"),
    "edge_reciprocal": _flag("the importer also sells to the exporter", "no trade in the opposite direction"),
    "edge_in_cycle": _flag("the trade lies on a loop in the trade network", "the trade is not part of a loop"),
    "edge_in_cycle_same_hs": _flag("the same goods are traded round a loop that includes this trade", "no loop in these goods"),
    "cycle_len_same_hs": lambda v: f"the same goods pass round a loop of {int(v)} parties" if v >= 2 else "no loop in these goods",
    "exporter_resale_days_log": lambda v: f"the seller bought the same goods {math.expm1(v):.0f} days earlier" if math.expm1(v) < 360 else "the seller has not bought these goods before",
}


def describe(name: str, value: float) -> str:
    """Plain-English phrase for one indicator value."""
    fn = PHRASES.get(name)
    if fn is None:
        return f"{FEATURE_BY_NAME[name].description}: {value:.2f}" if name in FEATURE_BY_NAME else f"{name} = {value:.2f}"
    return fn(float(value))
