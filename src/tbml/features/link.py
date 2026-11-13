"""Cleaning and record linkage (Step 02, first half).

Raw records arrive as four tables held by different owners. This module cleans
them and links them into one row per invoice ("trade"), the unit that features,
rules and models work on.

Linkage keys mirror real practice: payments quote the invoice number, the
invoice quotes a bill-of-lading number, the customs declaration quotes both.
Whatever cannot be linked is kept and flagged as missing, because a missing
document is itself a risk indicator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tbml.data.generator import TradeDataset

RAW_TABLES = ["invoices", "shipments", "declarations", "payments"]


def clean_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Remove exact duplicate rows, normalise codes, drop unusable rows."""
    out: dict[str, pd.DataFrame] = {}
    for name in RAW_TABLES:
        out[name] = tables[name].drop_duplicates().reset_index(drop=True).copy()

    inv = out["invoices"]
    inv["hs_code"] = inv["hs_code"].astype(str).str.strip().str.zfill(6)
    inv["currency"] = inv["currency"].astype(str).str.strip().str.upper()
    inv["bl_id"] = inv["bl_id"].astype(str).str.strip().str.upper()
    inv = inv[(inv["quantity"] > 0) & (inv["unit_price"] > 0) & (inv["invoice_value"] > 0)]
    out["invoices"] = inv.drop_duplicates("invoice_id").reset_index(drop=True)

    ship = out["shipments"]
    ship["bl_id"] = ship["bl_id"].astype(str).str.strip().str.upper()
    ship["transshipment_ports"] = ship["transshipment_ports"].fillna("").astype(str)
    out["shipments"] = ship.drop_duplicates("bl_id").reset_index(drop=True)

    dec = out["declarations"]
    dec["declared_hs_code"] = dec["declared_hs_code"].astype(str).str.strip().str.zfill(6)
    out["declarations"] = dec.drop_duplicates("declaration_id").reset_index(drop=True)

    pay = out["payments"]
    pay["currency"] = pay["currency"].astype(str).str.strip().str.upper()
    out["payments"] = pay[pay["amount"] > 0].drop_duplicates("payment_id").reset_index(drop=True)
    return out


def _aggregate_payments(pay: pd.DataFrame, fx: dict[str, float]) -> pd.DataFrame:
    """One row per invoice: totals, timing and who paid."""
    pay = pay.sort_values(["invoice_id", "payment_date", "payment_id"], kind="stable").copy()
    pay["amount_usd"] = pay["amount"] * pay["currency"].map(fx)
    g = pay.groupby("invoice_id", sort=False)
    agg = g.agg(
        n_payments=("payment_id", "size"),
        paid_usd=("amount_usd", "sum"),
        first_payment_date=("payment_date", "min"),
        last_payment_date=("payment_date", "max"),
        n_distinct_payers=("payer_id", "nunique"),
        n_pay_currencies=("currency", "nunique"),
        payer_id=("payer_id", "first"),
        payer_bank_id=("payer_bank_id", "first"),
        payee_bank_id=("payee_bank_id", "first"),
        pay_currency=("currency", "first"),
        payers=("payer_id", lambda s: "|".join(pd.unique(s))),
    )
    return agg.reset_index()


def link_records(tables: dict[str, pd.DataFrame], fx: dict[str, float], clean: bool = True) -> pd.DataFrame:
    """Link invoices with shipments, customs declarations and payments.

    Returns one row per invoice. ``has_shipment`` / ``has_customs`` /
    ``has_payment`` tell which documents could be linked.
    """
    t = clean_tables(tables) if clean else tables
    inv = t["invoices"].rename(columns={"invoice_id": "trade_id"})
    inv["invoice_value_usd"] = inv["invoice_value"] * inv["currency"].map(fx)
    inv["unit_price_usd"] = inv["unit_price"] * inv["currency"].map(fx)

    ship = t["shipments"].drop(columns=["shipper_id", "consignee_id"], errors="ignore")
    linked = inv.merge(ship, on="bl_id", how="left", validate="many_to_one")
    linked["has_shipment"] = linked["ship_date"].notna().astype(int)

    # customs: by invoice number first, then by bill of lading
    dec = t["declarations"].drop(columns=["importer_id"], errors="ignore")
    dec_cols = ["declaration_id", "declaration_date", "declared_hs_code", "declared_value_usd", "declared_quantity", "declared_weight_kg", "origin_country"]
    by_invoice = dec.drop_duplicates("invoice_id").set_index("invoice_id")[dec_cols]
    by_bl = dec.drop_duplicates("bl_id").set_index("bl_id")[dec_cols]
    part_inv = by_invoice.reindex(linked["trade_id"]).reset_index(drop=True)
    part_bl = by_bl.reindex(linked["bl_id"]).reset_index(drop=True)
    found_inv = part_inv["declaration_id"].notna().to_numpy()
    found_bl = part_bl["declaration_id"].notna().to_numpy()
    customs = part_inv.where(pd.Series(found_inv), part_bl, axis=0)
    linked = pd.concat([linked.reset_index(drop=True), customs], axis=1)
    linked["has_customs"] = (found_inv | found_bl).astype(int)
    linked["customs_link"] = np.where(found_inv, "invoice", np.where(found_bl, "bl", ""))

    pay = _aggregate_payments(t["payments"], fx).rename(columns={"invoice_id": "trade_id"})
    linked = linked.merge(pay, on="trade_id", how="left", validate="one_to_one")
    linked["has_payment"] = linked["n_payments"].notna().astype(int)
    linked["n_payments"] = linked["n_payments"].fillna(0).astype(int)
    for col in ["n_distinct_payers", "n_pay_currencies"]:
        linked[col] = linked[col].fillna(0).astype(int)
    linked["paid_usd"] = linked["paid_usd"].fillna(0.0)
    for col in ["transshipment_ports", "customs_link", "payer_id", "payers", "payer_bank_id", "payee_bank_id", "pay_currency",
                "declared_hs_code", "declaration_id", "origin_country", "origin_port", "dest_port", "shipping_line_id"]:
        linked[col] = linked[col].fillna("").astype(str)
    return linked.sort_values(["invoice_date", "trade_id"], kind="stable").reset_index(drop=True)


def fx_dict(fx_rates: pd.DataFrame) -> dict[str, float]:
    return dict(zip(fx_rates["currency"], fx_rates["usd_per_unit"].astype(float)))


def link_dataset(ds: TradeDataset) -> pd.DataFrame:
    """Linked trades of a whole dataset (the pooled, centralised view)."""
    return link_records(ds.tables, fx_dict(ds.tables["fx_rates"]))


def write_joined_ground_truth(ds: TradeDataset, out_dir: str | Path) -> pd.DataFrame:
    """Joined trades plus labels: evaluation and centralised baselines only."""
    joined = link_dataset(ds).merge(ds.tables["labels"], on="trade_id", how="left", validate="one_to_one")
    path = Path(out_dir) / "ground_truth" / "joined_trades.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(path, index=False, date_format="%Y-%m-%d")
    return joined
