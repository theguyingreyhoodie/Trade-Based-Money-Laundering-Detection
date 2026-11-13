"""Synthetic data generator: schema, label rates, reproducibility, typology fingerprints."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tbml.data.generator import OWNER_TABLES, generate_dataset, load_dataset, write_dataset
from tbml.data.schema import REQUIRED_COLUMNS, TRUTH_COLUMNS, validate_dataset
from tbml.data.typologies import TYPOLOGIES
from tbml.features.link import link_dataset


def test_schema_and_integrity(dataset, cfg) -> None:
    report = validate_dataset(dataset, cfg, strict=False)
    failed = [c for c in report["checks"] if not c["ok"]]
    assert not failed, failed
    for name, cols in REQUIRED_COLUMNS.items():
        assert list(dataset.tables[name].columns[: len(cols)]) == cols or set(cols) <= set(dataset.tables[name].columns), name


def test_label_rates(dataset, cfg) -> None:
    labels = dataset.tables["labels"]
    rate = labels["is_illicit"].mean()
    assert 0.02 <= rate <= 0.055
    assert abs(rate - cfg.data.illicit_rate) < 0.01

    counts = labels.loc[labels["is_illicit"] == 1, "typology"].value_counts()
    assert set(counts.index) == set(TYPOLOGIES), "every typology must be injected"
    mix = cfg.data.typology_mix.to_dict()
    for name in TYPOLOGIES:
        share = counts[name] / counts.sum()
        assert abs(share - mix[name]) < 0.04, (name, share, mix[name])
    assert (labels.loc[labels["is_illicit"] == 0, "typology"] == "none").all()
    # a share of the cases is deliberately subtle, but not all of them
    subtle = labels.loc[labels["is_illicit"] == 1, "is_subtle"].mean()
    assert 0.10 < subtle < 0.50


def test_same_seed_reproduces_same_data(cfg, dataset) -> None:
    again = generate_dataset(cfg, 0)
    assert set(again.tables) == set(dataset.tables)
    for name, table in dataset.tables.items():
        pd.testing.assert_frame_equal(table, again.tables[name], check_exact=True, obj=name)


def test_different_seed_gives_different_data(cfg, dataset) -> None:
    other = generate_dataset(cfg, 1)
    assert not dataset.tables["invoices"]["invoice_value"].head(200).equals(other.tables["invoices"]["invoice_value"].head(200))
    assert abs(other.tables["labels"]["is_illicit"].mean() - cfg.data.illicit_rate) < 0.01


def test_write_and_load_round_trip(dataset, tmp_path) -> None:
    out = write_dataset(dataset, tmp_path / "ds")
    for owner, names in OWNER_TABLES.items():
        for name in names:
            assert (out / "raw" / owner / f"{name}.csv").is_file()
    loaded = load_dataset(out)
    assert loaded.seed == dataset.seed
    for name in ["invoices", "shipments", "payments", "declarations", "kyc_entities", "labels"]:
        a, b = dataset.tables[name], loaded.tables[name]
        assert list(a.columns) == list(b.columns), name
        assert len(a) == len(b), name
    # codes with leading zeros must survive the CSV round trip
    assert loaded.tables["invoices"]["hs_code"].str.strip().str.len().eq(6).all()
    assert loaded.tables["declarations"]["declared_hs_code"].str.len().eq(6).all()
    np.testing.assert_allclose(loaded.tables["invoices"]["invoice_value"], dataset.tables["invoices"]["invoice_value"])


def test_owner_views_hold_no_ground_truth(dataset) -> None:
    for owner in OWNER_TABLES:
        for name, table in dataset.owner_view(owner).items():
            assert not (set(table.columns) & TRUTH_COLUMNS), (owner, name)
    assert {"payments", "kyc_entities"} <= set(dataset.owner_view("bank"))
    assert {"invoices", "shipments"} <= set(dataset.owner_view("trade"))
    assert {"declarations"} <= set(dataset.owner_view("customs"))


def _rank_auc(score: np.ndarray, y: np.ndarray) -> float:
    order = pd.Series(score).rank().to_numpy()
    pos = y == 1
    return float((order[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def test_identifiers_do_not_leak_the_label(dataset) -> None:
    """Sequence numbers inside ids must carry no information about being illicit."""
    labels = dataset.tables["labels"].set_index("trade_id")
    inv = dataset.tables["invoices"].drop_duplicates("invoice_id")
    y = labels.loc[inv["invoice_id"], "is_illicit"].to_numpy()
    bl_number = inv["bl_id"].str.extract(r"(\d+)")[0].astype(float).to_numpy()
    assert abs(_rank_auc(bl_number, y) - 0.5) < 0.08

    truth = dataset.tables["entity_truth"]
    shell = truth["is_shell"].to_numpy()
    entity_number = truth["entity_id"].str.extract(r"(\d+)")[0].astype(float).to_numpy()
    assert abs(_rank_auc(entity_number, shell) - 0.5) < 0.12
    kyc = dataset.tables["kyc_entities"]
    assert kyc["entity_id"].str.match(r"^ENT-\d{5}$").all()
    assert kyc["address_id"].str.match(r"^ADR-\d{6}$").all()
    assert kyc["director_id"].str.match(r"^DIR-\d{6}$").all()


@pytest.fixture(scope="module")
def joined(dataset) -> pd.DataFrame:
    ref = dataset.tables["hs_reference"].set_index("hs_code")
    j = link_dataset(dataset).merge(dataset.tables["labels"], on="trade_id")
    j["price_ratio"] = j["unit_price_usd"] / ref["ref_price_usd"].reindex(j["hs_code"]).to_numpy()
    return j


def test_typology_fingerprints(joined) -> None:
    """Each typology leaves the trace it is defined by (on average, not in every case)."""
    j = joined
    legit = j[j["typology"] == "none"]

    def rows(name: str) -> pd.DataFrame:
        return j[j["typology"] == name]

    assert rows("over_invoicing")["price_ratio"].median() > 1.5 * legit["price_ratio"].median()
    assert rows("under_invoicing")["price_ratio"].median() < 0.75 * legit["price_ratio"].median()

    multi = rows("multiple_invoicing")
    per_bl = j.groupby("bl_id")["trade_id"].transform("size")
    assert (per_bl[multi.index] >= 2).mean() > 0.95, "duplicate invoices share one bill of lading"

    phantom = rows("phantom_shipment")
    # expected: 15% forged complete papers, 25% near-empty container, 60% nothing (about 30 cases here)
    assert phantom["has_customs"].mean() < 0.40
    assert phantom["has_shipment"].mean() < 0.75
    assert legit["has_customs"].mean() > 0.97 and legit["has_shipment"].mean() > 0.97
    assert phantom["has_payment"].all(), "a phantom shipment is still paid for"

    ship = rows("over_under_shipment")
    ratio = (ship["shipped_quantity"] / ship["quantity"]).dropna()
    assert ((ratio < 0.92) | (ratio > 1.08)).mean() > 0.9

    mis = rows("misdescription")
    assert (mis["declared_hs_code"] != mis["hs_code"]).mean() > 0.95
    assert (legit["declared_hs_code"] != legit["hs_code"])[legit["has_customs"] == 1].mean() < 0.05

    third = rows("third_party_payment")
    assert (third["payer_id"] != third["importer_id"]).mean() > 0.95
    assert (legit["payer_id"] != legit["importer_id"]).mean() < 0.08

    routed = rows("high_risk_routing")
    assert (routed["transshipment_ports"] != "").mean() > 0.95
    assert (legit["transshipment_ports"] != "").mean() < 0.40


def test_circular_trades_form_cycles(dataset, joined) -> None:
    import networkx as nx

    circ = joined[joined["typology"] == "circular_trading"]
    assert len(circ) >= 3
    graph = nx.DiGraph()
    graph.add_edges_from(zip(circ["exporter_id"], circ["importer_id"]))
    on_cycle = {node for comp in nx.strongly_connected_components(graph) if len(comp) > 1 for node in comp}
    share = np.mean([(u in on_cycle) and (v in on_cycle) for u, v in zip(circ["exporter_id"], circ["importer_id"])])
    assert share > 0.8


def test_shells_are_young_and_benign_lookalikes_exist(dataset, cfg) -> None:
    kyc = dataset.tables["kyc_entities"].set_index("entity_id")
    truth = dataset.tables["entity_truth"].set_index("entity_id")
    start = pd.Timestamp(cfg.data.start_date)
    age = (start - kyc["registration_date"]).dt.days
    shells = truth["is_shell"] == 1
    assert age[shells].median() < age[~shells].median()
    # benign firms that look like shells: founded during the period, or at a shared address
    assert (age[~shells] < 0).sum() > 0
    shared = kyc["address_id"].map(kyc["address_id"].value_counts()) > 1
    assert (shared & ~shells).sum() >= 10, "benign firms at shared addresses"
    # benign noise that mimics typologies must be present among legitimate trades
    j = link_dataset(dataset).merge(dataset.tables["labels"], on="trade_id")
    legit = j[j["is_illicit"] == 0]
    assert (legit["has_customs"] == 0).sum() > 0
    assert (legit["payer_id"] != legit["importer_id"]).sum() > 0
    assert (legit["declared_hs_code"] != legit["hs_code"])[legit["has_customs"] == 1].sum() > 0
    assert (legit.groupby("bl_id")["trade_id"].transform("size") > 1).sum() > 0, "benign split invoicing"
