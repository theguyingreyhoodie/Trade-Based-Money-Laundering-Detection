"""TBML typologies: which trades are illicit and how their documents are manipulated.

Two stages:

1. ``illicit_specs`` decides who trades what and when for every illicit trade
   (parties come from laundering networks).
2. ``apply_typologies`` mutates the primitives of those trades so that the
   resulting documents carry the fingerprint of the typology.

A configurable share of cases is "subtle": the manipulation is small enough to
overlap with benign noise, so no typology is perfectly separable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tbml.config import Config
from tbml.data.entities import Entities, Network
from tbml.data.legit import SPEC_COLUMNS, sibling_hs
from tbml.data.reference import World

TYPOLOGIES = [
    "over_invoicing",
    "under_invoicing",
    "multiple_invoicing",
    "phantom_shipment",
    "over_under_shipment",
    "misdescription",
    "third_party_payment",
    "circular_trading",
    "shell_company",
    "high_risk_routing",
]

#: label "none" (legitimate) is class 0; typologies follow in the order above
TYPOLOGY_CLASSES = ["none", *TYPOLOGIES]


class _Picker:
    """Draws parties, goods and dates for illicit trades.

    Most illicit trades run through recruited, established firms and often
    through one of their existing trading relationships, so that an illicit
    trade is not identifiable from who the parties are alone.
    """

    def __init__(self, rng: np.random.Generator, world: World, ents: Entities, relationships: pd.DataFrame, cfg: Config) -> None:
        self.rng = rng
        self.world = world
        self.comp = ents.companies
        self.end = pd.Timestamp(cfg.data.end_date)
        modes = cfg.data.networks.pair_modes.to_dict()
        self.modes = list(modes.keys())
        self.mode_p = np.array(list(modes.values()), dtype=float) / sum(modes.values())
        ordinary = self.comp[(self.comp["entity_type"] == "company") & (~self.comp["is_shell"])]
        imp = ordinary[ordinary["role"].isin(["importer", "trader"])]
        exp = ordinary[ordinary["role"].isin(["exporter", "trader"])]
        self.legit_importers = imp.index.to_numpy()
        self.legit_importer_p = imp["size"].to_numpy() / imp["size"].sum()
        self.legit_exporters = exp.index.to_numpy()
        self.legit_exporter_p = exp["size"].to_numpy() / exp["size"].sum()
        self.buyers_of: dict[str, list[tuple[str, str, float]]] = {}
        self.suppliers_of: dict[str, list[tuple[str, str, float]]] = {}
        for r in relationships.itertuples():
            self.buyers_of.setdefault(r.exporter_id, []).append((r.importer_id, r.hs_code, float(r.pair_effect)))
            self.suppliers_of.setdefault(r.importer_id, []).append((r.exporter_id, r.hs_code, float(r.pair_effect)))

    def can_export(self, eid: str) -> bool:
        return self.comp.at[eid, "role"] in ("exporter", "trader")

    def can_import(self, eid: str) -> bool:
        return self.comp.at[eid, "role"] in ("importer", "trader")

    def country(self, eid: str) -> str:
        return str(self.comp.at[eid, "country"])

    def pair(self, net: Network, typology: str) -> tuple[str, str, str, float] | None:
        """Draw (exporter, importer, HS code, pair price effect); None if impossible."""
        mode = "shell" if typology == "shell_company" else str(self.rng.choice(self.modes, p=self.mode_p))
        for _ in range(40):
            if mode == "partner":
                drawn = self._partner(net)
            elif mode == "internal":
                drawn = self._internal(net)
            else:
                drawn = self._shell(net)
            if drawn is None:  # fall back towards modes that are always possible
                mode = "internal" if mode == "partner" else "shell"
                continue
            exp_id, imp_id, hs, effect = drawn
            if exp_id != imp_id and self.country(exp_id) != self.country(imp_id):
                return exp_id, imp_id, hs, effect
        return None

    def _partner(self, net: Network) -> tuple[str, str, str, float] | None:
        """A recruited firm trades with one of its ordinary, existing partners."""
        if not net.complicit:
            return None
        firm = str(self.rng.choice(net.complicit))
        options = []
        if self.buyers_of.get(firm):
            options.append("sell")
        if self.suppliers_of.get(firm):
            options.append("buy")
        if not options:
            return None
        if self.rng.choice(options) == "sell":
            other, hs, effect = self.buyers_of[firm][int(self.rng.integers(len(self.buyers_of[firm])))]
            return firm, other, hs, effect
        other, hs, effect = self.suppliers_of[firm][int(self.rng.integers(len(self.suppliers_of[firm])))]
        return other, firm, hs, effect

    def _internal(self, net: Network) -> tuple[str, str, str, float] | None:
        """Two recruited firms of the same network (shells only if needed)."""
        exporters = [m for m in net.complicit if self.can_export(m)] or [m for m in net.shells if self.can_export(m)]
        importers = [m for m in net.complicit if self.can_import(m)] or [m for m in net.shells if self.can_import(m)]
        if not exporters or not importers:
            return None
        exp_id = str(self.rng.choice(exporters))
        imp_id = str(self.rng.choice(importers))
        return exp_id, imp_id, self.hs(net, exp_id), float(self.rng.normal(0.0, 0.6))

    def _shell(self, net: Network) -> tuple[str, str, str, float] | None:
        """A shell on one side; the other side is a recruited firm or an ordinary company."""
        shell = str(self.rng.choice(net.shells))
        sells = self.can_export(shell) and (not self.can_import(shell) or self.rng.random() < 0.5)
        if sells:
            members = [m for m in net.complicit if self.can_import(m)]
            other = str(self.rng.choice(members)) if members and self.rng.random() < 0.6 else str(self.rng.choice(self.legit_importers, p=self.legit_importer_p))
            return shell, other, self.hs(net, shell), float(self.rng.normal(0.0, 0.6))
        members = [m for m in net.complicit if self.can_export(m)]
        other = str(self.rng.choice(members)) if members and self.rng.random() < 0.6 else str(self.rng.choice(self.legit_exporters, p=self.legit_exporter_p))
        return other, shell, self.hs(net, other), float(self.rng.normal(0.0, 0.6))

    def hs(self, net: Network, exporter: str) -> str:
        products = self.comp.at[exporter, "products"]
        if products and self.rng.random() < 0.5:
            return str(self.rng.choice(products))
        return str(self.rng.choice(net.hs_codes))

    def date(self, net: Network, *entities: str, tail_days: int = 0) -> pd.Timestamp | None:
        """Uniform date inside the network's active window, after all parties exist."""
        lo = max([net.start, *[self.comp.at[e, "registration_date"] + pd.Timedelta(days=15) for e in entities]])
        hi = min(net.end, self.end) - pd.Timedelta(days=tail_days)
        if lo >= hi:
            return None
        return (lo + pd.Timedelta(days=int(self.rng.integers(0, (hi - lo).days + 1)))).normalize()


def _row(exp_id: str, imp_id: str, hs: str, date: pd.Timestamp, typology: str, net: Network, effect: float, **extra) -> dict:
    row = {
        "exporter_id": exp_id,
        "importer_id": imp_id,
        "hs_code": hs,
        "invoice_date": date,
        "typology": typology,
        "network_id": net.network_id,
        "pair_effect": effect,
        "qty_override": np.nan,
        "price_usd_override": np.nan,
        "group_key": "",
        "dup_of": -1,
        "payer_override": "",
    }
    row.update(extra)
    return row


def _pick_network(rng: np.random.Generator, nets: list[Network]) -> Network:
    w = np.array([(n.end - n.start).days + 30 for n in nets], dtype=float)
    return nets[int(rng.choice(len(nets), p=w / w.sum()))]


def illicit_specs(rng: np.random.Generator, world: World, ents: Entities, relationships: pd.DataFrame, cfg: Config, n_illicit: int) -> pd.DataFrame:
    """Specs of all illicit trades. ``dup_of`` / ``group_key`` tie grouped schemes together."""
    mix = cfg.data.typology_mix.to_dict()
    tcfg = cfg.data.typologies
    total = sum(mix.values())
    picker = _Picker(rng, world, ents, relationships, cfg)
    rows: list[dict] = []

    for typology in TYPOLOGIES:
        target = int(round(n_illicit * mix.get(typology, 0.0) / total))
        nets = [n for n in ents.networks if typology in n.typologies]
        if target <= 0 or not nets:
            continue
        made = 0
        attempts = 0
        while made < target and attempts < target * 30:
            attempts += 1
            net = _pick_network(rng, nets)

            if typology == "circular_trading":
                p = tcfg.circular_trading
                members = net.members
                length = int(min(len(members), rng.integers(p.loop_len[0], p.loop_len[1] + 1)))
                if length < 3:
                    continue
                loop = None
                for _ in range(30):
                    cand = list(rng.choice(members, size=length, replace=False))
                    countries = [picker.country(e) for e in cand]
                    if all(countries[j] != countries[(j + 1) % length] for j in range(length)):
                        loop = cand
                        break
                if loop is None:
                    continue
                span = int(p.hop_days[1]) * length
                d0 = picker.date(net, *loop, tail_days=span)
                if d0 is None:
                    continue
                hs = str(rng.choice(net.hs_codes))
                ref = world.hs.loc[hs]
                price = ref["ref_price_usd"] * float(np.exp(rng.normal(0.0, 0.3 * ref["ref_log_sigma"])))
                qty = ref["typical_qty"] * float(rng.lognormal(0.0, 0.5))
                key = f"LOOP-{len(rows):06d}"
                date = d0
                for j in range(length):
                    rows.append(
                        _row(loop[j], loop[(j + 1) % length], hs, date, typology, net, 0.0,
                             qty_override=qty * float(rng.uniform(0.97, 1.03)), price_usd_override=price, group_key=key)
                    )
                    price *= 1.0 + float(rng.uniform(p.markup[0], p.markup[1]))
                    date = date + pd.Timedelta(days=int(rng.integers(p.hop_days[0], p.hop_days[1] + 1)))
                made += length
                continue

            pair = picker.pair(net, typology)
            if pair is None:
                continue
            exp_id, imp_id, hs, effect = pair

            if typology == "multiple_invoicing":
                p = tcfg.multiple_invoicing
                date = picker.date(net, exp_id, imp_id, tail_days=int(p.max_delay_days))
                if date is None:
                    continue
                key = f"MULTI-{len(rows):06d}"
                base_idx = len(rows)
                rows.append(_row(exp_id, imp_id, hs, date, typology, net, effect, group_key=key))
                n_extra = 2 if rng.random() < p.extra_invoices_p2 else 1
                for _ in range(n_extra):
                    later = date + pd.Timedelta(days=int(rng.integers(0, int(p.max_delay_days) + 1)))
                    rows.append(_row(exp_id, imp_id, hs, later, typology, net, effect, group_key=key, dup_of=base_idx))
                made += 1 + n_extra
                continue

            date = picker.date(net, exp_id, imp_id)
            if date is None:
                continue
            extra = {}
            if typology == "third_party_payment":
                vehicles = list(net.payers)
                if rng.random() < 0.25:  # sometimes a shell of the network pays instead of a dedicated vehicle
                    vehicles = [x for x in net.shells if x not in (exp_id, imp_id)] or vehicles
                if not vehicles:
                    continue
                extra["payer_override"] = str(rng.choice(vehicles))
            rows.append(_row(exp_id, imp_id, hs, date, typology, net, effect, **extra))
            made += 1

    spec = pd.DataFrame(rows, columns=SPEC_COLUMNS)
    spec["invoice_date"] = pd.to_datetime(spec["invoice_date"])
    return spec


def _other_bank(rng: np.random.Generator, banks: pd.DataFrame, country: str, region_countries: np.ndarray, current: str) -> str:
    local = [b for b in banks.index[banks["country"] == country] if b != current]
    if local:
        return str(rng.choice(local))
    regional = [b for b in banks.index[banks["country"].isin(region_countries)] if b != current]
    return str(rng.choice(regional)) if regional else current


def apply_typologies(rng: np.random.Generator, P: pd.DataFrame, world: World, ents: Entities, cfg: Config) -> pd.DataFrame:
    """Mutate the primitives of illicit trades according to their typology."""
    tcfg = cfg.data.typologies
    comp = ents.companies
    banks = ents.banks
    subtle_all = rng.random(len(P)) < tcfg.subtle_share
    P["is_subtle"] = False
    codes = world.country_codes
    listed_ports = np.array([f"{c}01" for c in codes[world.countries["risk_tier"].to_numpy() == "high"]])
    freezone_ports = np.array([f"{c}01" for c in codes[world.countries["risk_tier"].to_numpy() == "offshore"]])
    offshore_banks = banks.index[banks["country"].isin(codes[world.countries["risk_tier"].to_numpy() == "offshore"])].to_numpy()

    def rows_of(name: str) -> np.ndarray:
        return np.flatnonzero(P["typology"].to_numpy() == name)

    def mark_subtle(idx: np.ndarray) -> np.ndarray:
        s = subtle_all[idx]
        P.loc[idx[s], "is_subtle"] = True
        return s

    # ---- over-invoicing: inflated unit price moves value to the exporter
    idx = rows_of("over_invoicing")
    if len(idx):
        p = tcfg.over_invoicing
        s = mark_subtle(idx)
        f = np.clip(rng.lognormal(np.log(p.factor_median), p.factor_sigma, len(idx)), 1.4, 8.0)
        f[s] = rng.uniform(p.subtle_range[0], p.subtle_range[1], int(s.sum()))
        P.loc[idx, "unit_price_usd"] *= f
        true_value = rng.random(len(idx)) < p.customs_true_value_p  # customs still sees the real value
        P.loc[idx[true_value], "decl_value_ratio"] /= f[true_value]

    # ---- under-invoicing: deflated unit price moves value to the importer
    idx = rows_of("under_invoicing")
    if len(idx):
        p = tcfg.under_invoicing
        s = mark_subtle(idx)
        f = np.clip(rng.lognormal(np.log(p.factor_median), p.factor_sigma, len(idx)), 0.10, 0.72)
        f[s] = rng.uniform(p.subtle_range[0], p.subtle_range[1], int(s.sum()))
        P.loc[idx, "unit_price_usd"] *= f
        true_value = rng.random(len(idx)) < 0.10
        P.loc[idx[true_value], "decl_value_ratio"] /= f[true_value]

    # ---- multiple invoicing: several invoices (and payments) for one shipment
    idx = rows_of("multiple_invoicing")
    if len(idx):
        p = tcfg.multiple_invoicing
        P.loc[idx, "bl_ref"] = P.loc[idx, "bl_id"]  # the B/L must look valid to every financing bank
        P.loc[idx, ["has_shipment", "has_customs"]] = True
        shipment_cols = ["bl_id", "bl_ref", "shipping_line_id", "origin_port", "dest_port", "via_ports", "ship_lag_days",
                         "transit_noise_days", "ship_qty_ratio", "ship_qty_typo", "weight_hs", "weight_factor"]
        region_of = world.countries["region"]
        for i in idx:
            base = int(P.at[i, "dup_of"])
            if base < 0:
                continue
            for col in shipment_cols:
                P.at[i, col] = P.at[base, col]
            P.at[i, "owns_shipment"] = False
            P.at[i, "owns_customs"] = False
            same = rng.random() < 0.6
            P.at[i, "qty"] = P.at[base, "qty"] if same else max(1, int(round(P.at[base, "qty"] * rng.uniform(0.95, 1.05))))
            P.at[i, "unit_price_usd"] = P.at[base, "unit_price_usd"] * (1.0 if same else rng.uniform(0.98, 1.02))
            P.at[i, "currency"] = P.at[base, "currency"]
            P.at[i, "pay_currency"] = P.at[base, "pay_currency"]
            if rng.random() < p.other_bank_p:  # financed through a different bank
                country = comp.at[P.at[i, "importer_id"], "country"]
                region_countries = region_of.index[region_of == region_of[country]].to_numpy()
                P.at[i, "payer_bank_id"] = _other_bank(rng, banks, country, region_countries, P.at[base, "payer_bank_id"])

    # ---- phantom shipment: invoice and payment exist, the goods do not
    idx = rows_of("phantom_shipment")
    if len(idx):
        p = tcfg.phantom_shipment
        mark_subtle(idx)
        u = rng.random(len(idx))
        fabricated = u < p.fabricated_docs_p  # complete set of forged papers: nothing is missing
        empty = (~fabricated) & (u < p.fabricated_docs_p + p.empty_container_p)
        absent = ~(fabricated | empty)
        P.loc[idx[absent], "has_shipment"] = False
        P.loc[idx[empty | fabricated], "has_shipment"] = True
        P.loc[idx[empty], "weight_factor"] *= rng.uniform(0.02, 0.15, int(empty.sum()))
        P.loc[idx[~fabricated], "has_customs"] = False
        P.loc[idx[fabricated], "has_customs"] = True
        P.loc[idx[fabricated], "is_subtle"] = True
        quick = rng.random(len(idx)) < 0.8
        P.loc[idx[quick], "pay_term"] = np.where(rng.random(int(quick.sum())) < 0.5, "advance", "sight")
        P.loc[idx[quick], "pay_lag_days"] = rng.integers(0, 12, int(quick.sum()))

    # ---- over-/under-shipment: shipped goods differ from the invoiced quantity
    idx = rows_of("over_under_shipment")
    if len(idx):
        p = tcfg.over_under_shipment
        s = mark_subtle(idx)
        under = rng.random(len(idx)) < p.under_p
        g = np.where(under, rng.uniform(p.under_range[0], p.under_range[1], len(idx)), rng.uniform(p.over_range[0], p.over_range[1], len(idx)))
        g_subtle = np.where(under, rng.uniform(p.subtle_under[0], p.subtle_under[1], len(idx)), rng.uniform(p.subtle_over[0], p.subtle_over[1], len(idx)))
        P.loc[idx, "ship_qty_ratio"] = np.where(s, g_subtle, g)
        P.loc[idx, ["has_shipment", "has_customs"]] = True

    # ---- misdescription: the goods declared to customs are not the goods invoiced
    idx = rows_of("misdescription")
    if len(idx):
        p = tcfg.misdescription
        s = mark_subtle(idx)
        hs_codes = world.hs.index.to_numpy()
        sector = world.hs["sector"]
        for j, i in enumerate(idx):
            invoice_hs = P.at[i, "hs_code"]
            if s[j]:  # sibling code plus modest weight and price anomalies
                P.at[i, "decl_hs_code"] = sibling_hs(invoice_hs, rng)
                P.at[i, "weight_factor"] *= rng.uniform(0.55, 0.80) if rng.random() < 0.5 else rng.uniform(1.30, 1.70)
                P.at[i, "unit_price_usd"] *= float(np.exp(rng.choice([-1, 1]) * rng.uniform(0.15, 0.30)))
            else:
                cross = rng.random() < p.cross_chapter_p
                pool = hs_codes[(sector != sector[invoice_hs]).to_numpy()] if cross else hs_codes[((sector == sector[invoice_hs]) & (hs_codes != invoice_hs)).to_numpy()]
                true_hs = str(rng.choice(pool)) if len(pool) else sibling_hs(invoice_hs, rng)
                P.at[i, "decl_hs_code"] = true_hs
                if true_hs in world.hs.index:
                    P.at[i, "weight_hs"] = true_hs
                if rng.random() < 0.4:
                    P.at[i, "decl_value_ratio"] *= float(rng.lognormal(0.0, 0.5))
        P.loc[idx, ["has_shipment", "has_customs"]] = True

    # ---- third-party payment: settled by an unrelated vehicle, often offshore
    idx = rows_of("third_party_payment")
    if len(idx):
        p = tcfg.third_party_payment
        mark_subtle(idx)
        nets = {n.network_id: n for n in ents.networks}
        for i in idx:
            payer = P.at[i, "payer_override"]
            P.at[i, "payer_id"] = payer
            P.at[i, "payer_bank_id"] = comp.at[payer, "home_bank_id"]
            if rng.random() < p.currency_switch_p:
                P.at[i, "pay_currency"] = "USD" if P.at[i, "currency"] != "USD" else str(rng.choice(["EUR", "AED"]))
            if rng.random() < p.split_p:
                P.at[i, "n_payments"] = 2
                others = [x for x in nets[P.at[i, "network_id"]].payers if x != payer]
                P.at[i, "payer2_id"] = str(rng.choice(others)) if others else payer

    # ---- circular trading: the same goods go round a loop (parties and prices set in the spec)
    idx = rows_of("circular_trading")
    if len(idx):
        mark_subtle(idx)
        P.loc[idx, "pay_term"] = "sight"
        P.loc[idx, "pay_lag_days"] = rng.integers(1, 8, len(idx))
        P.loc[idx, "ship_lag_days"] = rng.integers(1, 5, len(idx))

    # ---- shell company: pass-through trades of a young, thinly documented vehicle
    idx = rows_of("shell_company")
    if len(idx):
        p = tcfg.shell_company
        mark_subtle(idx)
        rnd = rng.random(len(idx)) < p.round_amount_p
        P.loc[idx[rnd], "round_base"] = rng.choice([5000, 10000], int(rnd.sum()))
        P.loc[idx, "unit_price_usd"] *= np.exp(rng.uniform(-p.price_dev, p.price_dev, len(idx)))
        P.loc[idx, "qty"] = np.maximum(1, np.round(P.loc[idx, "qty"].to_numpy() * rng.uniform(1.5, 4.0, len(idx)))).astype(np.int64)
        adv = rng.random(len(idx)) < 0.5
        P.loc[idx[adv], "pay_term"] = "advance"
        P.loc[idx[adv], "pay_lag_days"] = rng.integers(0, 4, int(adv.sum()))
        off = rng.random(len(idx)) < 0.3
        if len(offshore_banks):
            P.loc[idx[off], "payee_bank_id"] = rng.choice(offshore_banks, int(off.sum()))

    # ---- high-risk routing: needless stopovers in high-risk / free-zone jurisdictions
    idx = rows_of("high_risk_routing")
    if len(idx):
        p = tcfg.high_risk_routing
        mark_subtle(idx)
        for i in idx:
            banned = {P.at[i, "origin_port"], P.at[i, "dest_port"]}
            k = int(rng.integers(p.n_hops[0], p.n_hops[1] + 1))
            stops: list[str] = []
            for _ in range(k):
                source = listed_ports if rng.random() < p.listed_stop_p else freezone_ports
                pool = [x for x in source if x not in banned and x not in stops]
                if pool:
                    stops.append(str(rng.choice(pool)))
            P.at[i, "via_ports"] = "|".join(stops)
            if len(offshore_banks) and rng.random() < p.offshore_bank_p:
                P.at[i, "payee_bank_id"] = str(rng.choice(offshore_banks))
            if rng.random() < p.misprice_p:
                P.at[i, "unit_price_usd"] *= float(np.exp(rng.choice([-1, 1]) * rng.uniform(0.15, p.price_dev)))
        P.loc[idx, "has_shipment"] = True
    return P
