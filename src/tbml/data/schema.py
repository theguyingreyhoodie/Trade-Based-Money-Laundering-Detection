"""Schema and integrity checks for a generated dataset."""

from __future__ import annotations

from typing import Any

import pandas as pd

from tbml.config import Config
from tbml.data.generator import OWNER_TABLES, TradeDataset
from tbml.data.typologies import TYPOLOGIES

REQUIRED_COLUMNS = {
    "invoices": ["invoice_id", "invoice_date", "exporter_id", "importer_id", "hs_code", "goods_description", "quantity", "unit",
                 "unit_price", "currency", "invoice_value", "bl_id", "payment_terms"],
    "shipments": ["bl_id", "shipping_line_id", "shipper_id", "consignee_id", "origin_port", "dest_port", "transshipment_ports",
                  "ship_date", "arrival_date", "shipped_quantity", "gross_weight_kg", "container_count"],
    "payments": ["payment_id", "invoice_id", "payment_date", "payer_id", "payee_id", "amount", "currency", "payer_bank_id", "payee_bank_id"],
    "declarations": ["declaration_id", "invoice_id", "bl_id", "declaration_date", "importer_id", "declared_hs_code",
                     "declared_value_usd", "declared_quantity", "declared_weight_kg", "origin_country"],
    "kyc_entities": ["entity_id", "name", "entity_type", "country", "registration_date", "address_id", "director_id", "home_bank_id"],
    "banks": ["bank_id", "bank_name", "country"],
    "shipping_lines": ["shipping_line_id", "line_name"],
    "hs_reference": ["hs_code", "description", "chapter", "sector", "unit", "ref_price_usd", "ref_log_sigma", "kg_per_unit"],
    "countries": ["country", "name", "region", "risk_tier", "fatf_listed", "is_ftz", "is_hub", "currency", "risk_score"],
    "ports": ["port", "port_name", "country", "lat", "lon", "is_hub"],
    "fx_rates": ["currency", "usd_per_unit"],
    "labels": ["trade_id", "is_illicit", "typology", "network_id", "is_subtle"],
    "entity_truth": ["entity_id", "is_shell", "is_complicit", "is_payer", "network_id"],
}

PRIMARY_KEYS = {"invoices": "invoice_id", "shipments": "bl_id", "payments": "payment_id", "declarations": "declaration_id",
                "kyc_entities": "entity_id", "labels": "trade_id"}

#: ground-truth columns that must never appear in a data owner's view
TRUTH_COLUMNS = {"is_illicit", "typology", "network_id", "is_subtle", "is_shell", "is_complicit", "is_payer", "role", "sectors", "products", "size"}


def validate_dataset(ds: TradeDataset, cfg: Config, strict: bool = True) -> dict[str, Any]:
    """Run all checks; raise ``AssertionError`` on the first failure when ``strict``."""
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        if strict and not ok:
            raise AssertionError(f"dataset check failed: {name} {detail}")

    t = ds.tables
    for name, cols in REQUIRED_COLUMNS.items():
        missing = [c for c in cols if c not in t[name].columns]
        check(f"columns:{name}", not missing, f"missing {missing}" if missing else "")

    for name, key in PRIMARY_KEYS.items():
        deduped = t[name].drop_duplicates()
        check(f"unique:{name}.{key}", deduped[key].is_unique, "after removing exact duplicate rows")

    for owner, names in OWNER_TABLES.items():
        leaked = sorted(set().union(*[set(t[n].columns) for n in names]) & TRUTH_COLUMNS)
        check(f"no_truth_columns:{owner}", not leaked, f"leaked {leaked}" if leaked else "")

    inv = t["invoices"].drop_duplicates()
    entities = set(t["kyc_entities"]["entity_id"])
    hs_codes = set(t["hs_reference"]["hs_code"])
    ports = set(t["ports"]["port"])
    invoice_ids = set(inv["invoice_id"])
    check("fk:payments.invoice_id", set(t["payments"]["invoice_id"]) <= invoice_ids)
    check("fk:declarations.invoice_id", set(t["declarations"]["invoice_id"]) <= invoice_ids)
    check("fk:invoice parties", set(inv["exporter_id"]) | set(inv["importer_id"]) <= entities)
    check("fk:payment parties", set(t["payments"]["payer_id"]) | set(t["payments"]["payee_id"]) <= entities)
    check("fk:invoices.hs_code", set(inv["hs_code"].str.strip()) <= hs_codes)
    check("fk:shipment ports", set(t["shipments"]["origin_port"]) | set(t["shipments"]["dest_port"]) <= ports)
    check("fk:banks", set(t["payments"]["payer_bank_id"]) | set(t["payments"]["payee_bank_id"]) <= set(t["banks"]["bank_id"]))
    check("every invoice is paid", invoice_ids <= set(t["payments"]["invoice_id"]))
    check("exporter != importer", bool((inv["exporter_id"] != inv["importer_id"]).all()))

    check("positive:quantity", bool((inv["quantity"] > 0).all()))
    check("positive:unit_price", bool((inv["unit_price"] > 0).all()))
    check("positive:invoice_value", bool((inv["invoice_value"] > 0).all()))
    check("positive:payment amount", bool((t["payments"]["amount"] > 0).all()))
    start, end = pd.Timestamp(cfg.data.start_date), pd.Timestamp(cfg.data.end_date)
    check("dates:invoice within period", bool(inv["invoice_date"].between(start, end).all()))
    check("dates:arrival after departure", bool((t["shipments"]["arrival_date"] > t["shipments"]["ship_date"]).all()))

    labels = t["labels"]
    check("labels cover every invoice", set(labels["trade_id"]) == invoice_ids)
    rate = float(labels["is_illicit"].mean())
    check("illicit rate in 2-5%", 0.02 <= rate <= 0.05 or abs(rate - cfg.data.illicit_rate) < 0.01, f"rate={rate:.4f}")
    counts = labels.loc[labels["is_illicit"] == 1, "typology"].value_counts()
    absent = [name for name in TYPOLOGIES if counts.get(name, 0) == 0 and cfg.data.typology_mix.get(name, 0) > 0]
    check("every typology present", not absent, f"absent {absent}" if absent else "")
    check("legit rows labelled none", bool((labels.loc[labels["is_illicit"] == 0, "typology"] == "none").all()))

    return {
        "checks": checks,
        "n_failed": sum(not c["ok"] for c in checks),
        "illicit_rate": rate,
        "typology_rates": {name: float(counts.get(name, 0) / len(labels)) for name in TYPOLOGIES},
    }
