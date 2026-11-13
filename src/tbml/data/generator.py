"""Synthetic trade data generator (Step 01, data preparation).

``generate_dataset`` builds a world of entities and trades, injects ten TBML
typologies and benign noise, and returns the raw records as they would sit with
their owners (bank / trade / customs views) plus the ground truth.

Assumptions are documented in ``docs/data_generator.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tbml.config import Config
from tbml.data.entities import Entities, anonymise, attach_networks, build_entities
from tbml.data.legit import (
    apply_benign_noise,
    build_price_index,
    build_primitives,
    build_relationships,
    legit_specs,
    return_specs,
    ring_specs,
    spot_specs,
)
from tbml.data.reference import World, build_world, published_hs_reference, transit_days
from tbml.data.typologies import TYPOLOGIES, apply_typologies, illicit_specs
from tbml.utils import get_logger, load_json, save_json

log = get_logger(__name__)

GENERATOR_VERSION = "1.0"

#: raw tables grouped by the organisation that owns them
OWNER_TABLES = {
    "bank": ["payments", "kyc_entities", "banks"],
    "trade": ["invoices", "shipments", "shipping_lines"],
    "customs": ["declarations"],
}
REFERENCE_TABLES = ["hs_reference", "countries", "ports", "fx_rates"]
TRUTH_TABLES = ["labels", "entity_truth"]

#: columns that must be read back as strings / dates
DATE_COLUMNS = {
    "invoices": ["invoice_date"],
    "shipments": ["ship_date", "arrival_date"],
    "payments": ["payment_date"],
    "declarations": ["declaration_date"],
    "kyc_entities": ["registration_date"],
}
STRING_COLUMNS = {
    "invoices": ["invoice_id", "exporter_id", "importer_id", "hs_code", "goods_description", "unit", "currency", "bl_id", "payment_terms"],
    "shipments": ["bl_id", "shipping_line_id", "shipper_id", "consignee_id", "origin_port", "dest_port", "transshipment_ports"],
    "payments": ["payment_id", "invoice_id", "payer_id", "payee_id", "currency", "payer_bank_id", "payee_bank_id"],
    "declarations": ["declaration_id", "invoice_id", "bl_id", "importer_id", "declared_hs_code", "origin_country"],
    "kyc_entities": ["entity_id", "name", "entity_type", "country", "address_id", "director_id", "home_bank_id"],
    "banks": ["bank_id", "bank_name", "country"],
    "shipping_lines": ["shipping_line_id", "line_name"],
    "hs_reference": ["hs_code", "description", "chapter", "sector", "unit"],
    "countries": ["country", "name", "region", "risk_tier", "currency"],
    "ports": ["port", "port_name", "country"],
    "fx_rates": ["currency"],
    "labels": ["trade_id", "typology", "network_id"],
    "entity_truth": ["entity_id", "network_id"],
}


@dataclass
class TradeDataset:
    """All tables of one generated dataset."""

    seed: int
    tables: dict[str, pd.DataFrame]
    meta: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> pd.DataFrame:
        tables = self.__dict__.get("tables", {})
        if name in tables:
            return tables[name]
        raise AttributeError(name)

    def owner_view(self, owner: str) -> dict[str, pd.DataFrame]:
        """Raw tables held by one data owner (``bank``, ``trade`` or ``customs``)."""
        return {name: self.tables[name] for name in OWNER_TABLES[owner]}

    def reference(self) -> dict[str, pd.DataFrame]:
        return {name: self.tables[name] for name in REFERENCE_TABLES}


# ---------------------------------------------------------------------------
# primitives -> documents
# ---------------------------------------------------------------------------


def add_companions(rng: np.random.Generator, P: pd.DataFrame, rate: float) -> pd.DataFrame:
    """Benign split invoicing: one shipment covered by two invoices of the same parties.

    This is the legitimate look-alike of multiple invoicing. The two invoices
    together add up to the shipment, whereas duplicate invoices each claim all of it.
    """
    ok = (
        (P["typology"] == "none") & P["has_shipment"] & P["has_customs"] & (P["bl_ref"] == P["bl_id"]) & (P["qty"] >= 4) & (P["round_base"] == 0)
    ).to_numpy()
    chosen = np.flatnonzero(ok & (rng.random(len(P)) < rate))
    if not len(chosen):
        return P
    comp = P.iloc[chosen].copy()
    share = rng.uniform(0.4, 0.6, len(chosen))
    qty = P["qty"].to_numpy()[chosen]
    first = np.maximum(1, np.round(qty * share)).astype(np.int64)
    second = qty - first
    P.loc[chosen, "qty"] = first
    P.loc[chosen, "companion_qty"] = second
    comp["qty"] = second
    comp["owns_shipment"] = False
    comp["companion_qty"] = 0
    comp["n_payments"] = 1
    return pd.concat([P, comp], ignore_index=True)


def _route_metrics(P: pd.DataFrame, world: World) -> tuple[np.ndarray, np.ndarray]:
    """Route length (nm) and number of transshipments for every trade."""
    o = world.country_index(P["origin_port"].str[:3])
    d = world.country_index(P["dest_port"].str[:3])
    dist = world.distance_nm[o, d].copy()
    hops = np.zeros(len(P), dtype=np.int64)
    via = P["via_ports"].to_numpy()
    cidx = {c: i for i, c in enumerate(world.country_codes)}
    for i in np.flatnonzero(via != ""):
        stops = [cidx[p[:3]] for p in via[i].split("|")]
        path = [o[i], *stops, d[i]]
        dist[i] = sum(world.distance_nm[a, b] for a, b in zip(path[:-1], path[1:]))
        hops[i] = len(stops)
    return dist, hops


def finalise(P: pd.DataFrame, world: World, ents: Entities) -> dict[str, pd.DataFrame]:
    """Derive the four document tables from the primitives."""
    comp = ents.companies
    fx = P["currency"].map(world.fx).to_numpy()
    qty = P["qty"].to_numpy(dtype=np.int64)

    # ---- invoice amounts
    price_cur = P["unit_price_usd"].to_numpy() / fx
    price_cur = np.where(price_cur >= 1.0, np.round(price_cur, 4), np.round(price_cur, 6))
    value = np.round(qty * price_cur, 2)
    base = P["round_base"].to_numpy(dtype=float)
    lump = (base > 0) & (value >= 2 * base)
    value = np.where(lump, np.round(value / np.where(lump, base, 1.0)) * np.where(lump, base, 1.0), value)
    price_cur = np.where(lump, np.round(value / qty, 6), price_cur)
    value_usd = value * fx

    # ---- dates and route
    dist, hops = _route_metrics(P, world)
    transit = np.maximum(1, np.round(transit_days(dist, hops) + P["transit_noise_days"].to_numpy())).astype(int)
    invoice_date = pd.to_datetime(P["invoice_date"]).dt.normalize()
    ship_date = invoice_date + pd.to_timedelta(P["ship_lag_days"].to_numpy(), unit="D")
    arrival = ship_date + pd.to_timedelta(transit, unit="D")
    decl_date = arrival + pd.to_timedelta(P["decl_lag_days"].to_numpy(), unit="D")

    hs = world.hs
    unit = hs.loc[P["hs_code"], "unit"].to_numpy()
    kg = hs.loc[P["weight_hs"], "kg_per_unit"].to_numpy()
    ratio = P["ship_qty_ratio"].to_numpy()
    own_weight = qty * ratio * kg * P["weight_factor"].to_numpy()
    total_qty = (qty + P["companion_qty"].to_numpy()) * ratio
    total_weight = total_qty * kg * P["weight_factor"].to_numpy()

    invoices = pd.DataFrame(
        {
            "invoice_id": P["trade_id"].to_numpy(),
            "invoice_date": invoice_date.to_numpy(),
            "exporter_id": P["exporter_id"].to_numpy(),
            "importer_id": P["importer_id"].to_numpy(),
            "hs_code": P["hs_code"].to_numpy(),
            "goods_description": hs.loc[P["hs_code"], "description"].to_numpy(),
            "quantity": qty,
            "unit": unit,
            "unit_price": price_cur,
            "currency": P["currency"].to_numpy(),
            "invoice_value": value,
            "bl_id": P["bl_ref"].to_numpy(),
            "payment_terms": P["pay_term"].to_numpy(),
        }
    )

    s = (P["has_shipment"] & P["owns_shipment"]).to_numpy()
    shipments = pd.DataFrame(
        {
            "bl_id": P["bl_id"].to_numpy()[s],
            "shipping_line_id": P["shipping_line_id"].to_numpy()[s],
            "shipper_id": P["exporter_id"].to_numpy()[s],
            "consignee_id": P["importer_id"].to_numpy()[s],
            "origin_port": P["origin_port"].to_numpy()[s],
            "dest_port": P["dest_port"].to_numpy()[s],
            "transshipment_ports": P["via_ports"].to_numpy()[s],
            "ship_date": ship_date.to_numpy()[s],
            "arrival_date": arrival.to_numpy()[s],
            "shipped_quantity": np.maximum(0, np.round(total_qty * P["ship_qty_typo"].to_numpy())).astype(np.int64)[s],
            "gross_weight_kg": np.round(total_weight, 1)[s],
            "container_count": np.maximum(1, np.ceil(total_weight / 18000.0)).astype(np.int64)[s],
        }
    )

    c = (P["has_customs"] & P["owns_customs"]).to_numpy()
    declarations = pd.DataFrame(
        {
            "invoice_id": P["trade_id"].to_numpy()[c],
            "bl_id": P["bl_id"].to_numpy()[c],
            "declaration_date": decl_date.to_numpy()[c],
            "importer_id": P["importer_id"].to_numpy()[c],
            "declared_hs_code": P["decl_hs_code"].to_numpy()[c],
            "declared_value_usd": np.round(value_usd * P["decl_value_ratio"].to_numpy(), 2)[c],
            "declared_quantity": np.maximum(0, np.round(qty * ratio * P["decl_qty_ratio"].to_numpy())).astype(np.int64)[c],
            "declared_weight_kg": np.round(own_weight * P["decl_weight_ratio"].to_numpy(), 1)[c],
            "origin_country": P["origin_port"].str[:3].to_numpy()[c],
        }
    )
    declarations.insert(0, "declaration_id", [f"DEC-{i:08d}" for i in range(1, len(declarations) + 1)])

    # ---- payments (one row per instalment)
    term = P["pay_term"].to_numpy()
    anchor = np.where(term == "advance", invoice_date.to_numpy(), np.where(term == "sight", ship_date.to_numpy(), arrival.to_numpy()))
    first_pay = pd.to_datetime(anchor) + pd.to_timedelta(P["pay_lag_days"].to_numpy(), unit="D")
    pay_fx = P["pay_currency"].map(world.fx).to_numpy()
    total_pay = np.round(value * P["pay_ratio"].to_numpy() * fx / pay_fx, 2)
    n_pay = P["n_payments"].to_numpy()
    splits = {1: [1.0], 2: [0.5, 0.5], 3: [0.3, 0.3, 0.4]}
    payer2 = P["payer2_id"].to_numpy()
    parts = []
    for k in range(3):
        rows = np.flatnonzero(n_pay > k)
        if not len(rows):
            continue
        frac = np.array([splits[int(m)][k] for m in n_pay[rows]])
        amount = np.round(total_pay[rows] * frac, 2)
        last = n_pay[rows] == k + 1  # final instalment absorbs rounding
        prior = np.array([sum(splits[int(m)][:k]) for m in n_pay[rows]])
        amount = np.where(last, np.round(total_pay[rows] - np.round(total_pay[rows] * prior, 2), 2), amount)
        payer = P["payer_id"].to_numpy()[rows].copy()
        bank = P["payer_bank_id"].to_numpy()[rows].copy()
        if k == 1:
            alt = payer2[rows] != ""
            payer[alt] = payer2[rows][alt]
            bank[alt] = comp.loc[payer2[rows][alt], "home_bank_id"].to_numpy()
        parts.append(
            pd.DataFrame(
                {
                    "invoice_id": P["trade_id"].to_numpy()[rows],
                    "payment_date": (first_pay[rows] + pd.to_timedelta(k * P["pay_gap_days"].to_numpy()[rows], unit="D")).to_numpy(),
                    "payer_id": payer,
                    "payee_id": P["exporter_id"].to_numpy()[rows],
                    "amount": amount,
                    "currency": P["pay_currency"].to_numpy()[rows],
                    "payer_bank_id": bank,
                    "payee_bank_id": P["payee_bank_id"].to_numpy()[rows],
                }
            )
        )
    payments = pd.concat(parts, ignore_index=True).sort_values(["payment_date", "invoice_id"], kind="stable").reset_index(drop=True)
    payments.insert(0, "payment_id", [f"PAY-{i:08d}" for i in range(1, len(payments) + 1)])
    return {"invoices": invoices, "shipments": shipments, "declarations": declarations, "payments": payments}


def _mess_up(rng: np.random.Generator, tables: dict[str, pd.DataFrame], cfg: Config) -> None:
    """Raw-data blemishes that the cleaning step has to remove (no information change)."""
    ncfg = cfg.data.noise
    inv = tables["invoices"]
    messy = np.flatnonzero(rng.random(len(inv)) < ncfg.messy_code_rate)
    half = len(messy) // 2
    inv.loc[messy[:half], "hs_code"] = " " + inv.loc[messy[:half], "hs_code"] + " "
    inv.loc[messy[half:], "currency"] = inv.loc[messy[half:], "currency"].str.lower()
    for name in ["invoices", "shipments", "declarations", "payments"]:
        df = tables[name]
        dup = df.iloc[np.flatnonzero(rng.random(len(df)) < ncfg.duplicate_row_rate)]
        tables[name] = pd.concat([df, dup], ignore_index=True)


def kyc_view(companies: pd.DataFrame) -> pd.DataFrame:
    """Registry / KYC attributes of entities. Hidden truth columns are dropped."""
    cols = ["name", "entity_type", "country", "registration_date", "address_id", "director_id", "home_bank_id"]
    return companies[cols].reset_index()


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def generate_dataset(cfg: Config, seed: int | None = None) -> TradeDataset:
    """Generate one dataset. The same config and seed always give the same data."""
    seed = int(cfg.project.seed if seed is None else seed)
    rng = np.random.default_rng(seed)
    dcfg = cfg.data
    start = pd.Timestamp(dcfg.start_date)
    end = pd.Timestamp(dcfg.end_date)

    # world: ordinary firms and their relationships first, laundering networks on top
    world = build_world()
    ents = build_entities(rng, world, cfg)
    rel = build_relationships(rng, world, ents, cfg)
    ents = attach_networks(rng, world, ents, rel, cfg)
    ents, rel = anonymise(rng, ents, rel)
    n_months = (end.year - start.year) * 12 + end.month - start.month + 1
    price_index = build_price_index(rng, world, n_months, dcfg.legit.monthly_price_drift)

    # who trades what and when
    n_illicit = int(round(dcfg.n_trades * dcfg.illicit_rate))
    bad = illicit_specs(rng, world, ents, rel, cfg, n_illicit)
    n_legit = int(dcfg.n_trades) - len(bad)
    n_spot = int(round(n_legit * dcfg.legit.spot_rate))
    n_return = int(round(n_legit * dcfg.legit.return_rate))
    n_ring = int(round(n_legit * dcfg.legit.ring_trade_rate))
    regular = legit_specs(rng, rel, n_legit - n_spot - n_return - n_ring, end)
    good = pd.concat(
        [regular, spot_specs(rng, world, ents, cfg, n_spot), return_specs(rng, world, regular, cfg, n_return), ring_specs(rng, world, ents, cfg, n_ring)],
        ignore_index=True,
    )
    bad.loc[bad["dup_of"] >= 0, "dup_of"] += len(good)
    spec = pd.concat([good, bad], ignore_index=True)

    # documents: ordinary behaviour, benign noise, then the typology manipulations
    P = build_primitives(rng, spec, world, ents, cfg, price_index)
    P = apply_benign_noise(rng, P, cfg, world)
    P = apply_typologies(rng, P, world, ents, cfg)
    P = add_companions(rng, P, float(dcfg.legit.get("consolidated_rate", 0.0)))

    # chronological ids, so the identifier carries no information about the label
    order = np.lexsort((rng.random(len(P)), pd.to_datetime(P["invoice_date"]).to_numpy()))
    P = P.iloc[order].reset_index(drop=True)
    P["trade_id"] = [f"INV-{i:07d}" for i in range(1, len(P) + 1)]

    tables = finalise(P, world, ents)
    _mess_up(rng, tables, cfg)

    tables["kyc_entities"] = kyc_view(ents.companies)
    tables["banks"] = ents.banks.reset_index()
    tables["shipping_lines"] = ents.shipping_lines.reset_index()[["shipping_line_id", "line_name"]]
    tables["hs_reference"] = published_hs_reference(world, rng, dcfg.noise.reference_price_error)
    tables["countries"] = world.countries.reset_index()
    tables["ports"] = world.ports.reset_index()
    tables["fx_rates"] = pd.DataFrame({"currency": list(world.fx), "usd_per_unit": list(world.fx.values())})
    tables["labels"] = pd.DataFrame(
        {
            "trade_id": P["trade_id"],
            "is_illicit": (P["typology"] != "none").astype(int),
            "typology": P["typology"],
            "network_id": P["network_id"],
            "is_subtle": P["is_subtle"].astype(int),
        }
    )
    tables["entity_truth"] = ents.companies[["is_shell", "is_complicit", "is_payer", "network_id"]].astype(
        {"is_shell": int, "is_complicit": int, "is_payer": int}
    ).reset_index()

    labels = tables["labels"]
    counts = labels.loc[labels["is_illicit"] == 1, "typology"].value_counts()
    meta = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "n_trades": int(len(P)),
        "n_illicit": int(labels["is_illicit"].sum()),
        "illicit_rate": float(labels["is_illicit"].mean()),
        "typology_counts": {t: int(counts.get(t, 0)) for t in TYPOLOGIES},
        "n_entities": int(len(ents.companies)),
        "n_networks": len(ents.networks),
        "n_relationships": int(len(rel)),
        "period": [str(start.date()), str(end.date())],
        "rows": {name: int(len(df)) for name, df in tables.items()},
        "data_config": dcfg.to_dict(),
    }
    log.info("seed %d: %d trades, %d illicit (%.2f%%), %d entities", seed, meta["n_trades"], meta["n_illicit"], 100 * meta["illicit_rate"], meta["n_entities"])
    return TradeDataset(seed=seed, tables=tables, meta=meta)


def _folder_of(name: str) -> str:
    for owner, names in OWNER_TABLES.items():
        if name in names:
            return f"raw/{owner}"
    if name in REFERENCE_TABLES:
        return "reference"
    if name in TRUTH_TABLES:
        return "ground_truth"
    raise KeyError(name)


def write_dataset(ds: TradeDataset, out_dir: str | Path) -> Path:
    """Write every table as CSV under ``raw/<owner>/``, ``reference/`` and ``ground_truth/``."""
    out = Path(out_dir)
    for name, df in ds.tables.items():
        path = out / _folder_of(name) / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, date_format="%Y-%m-%d")
    save_json(ds.meta, out / "meta.json")
    return out


def read_table(path: str | Path, name: str) -> pd.DataFrame:
    """Read one table with stable dtypes (ids and codes stay strings)."""
    dtypes = {col: str for col in STRING_COLUMNS.get(name, [])}
    df = pd.read_csv(path, dtype=dtypes, keep_default_na=False, na_values=[""], parse_dates=DATE_COLUMNS.get(name, []))
    for col in STRING_COLUMNS.get(name, []):
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df


def load_dataset(data_dir: str | Path) -> TradeDataset:
    """Load a dataset written by :func:`write_dataset`."""
    root = Path(data_dir)
    tables = {}
    for name in [*sum(OWNER_TABLES.values(), []), *REFERENCE_TABLES, *TRUTH_TABLES]:
        tables[name] = read_table(root / _folder_of(name) / f"{name}.csv", name)
    meta = load_json(root / "meta.json")
    return TradeDataset(seed=int(meta["seed"]), tables=tables, meta=meta)
