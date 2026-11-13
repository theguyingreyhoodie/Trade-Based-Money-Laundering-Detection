"""Cleaning, linkage, split and risk indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tbml.data.typologies import TYPOLOGIES, TYPOLOGY_CLASSES
from tbml.features.build import FeatureSet, Reference, build_features, row_features
from tbml.features.dictionary import CONTEXT_FEATURES, FEATURE_BY_NAME, FEATURE_NAMES, FEATURES, GROUPS, ROW_FEATURES, features_markdown
from tbml.features.link import clean_tables, fx_dict, link_dataset, link_records
from tbml.features.pipeline import feature_report, rank_auc
from tbml.models.split import assign_split, check_stratification, split_cutoffs


# --------------------------------------------------------------------------- cleaning + linkage


def test_cleaning_removes_blemishes(dataset) -> None:
    raw = dataset.tables
    clean = clean_tables(raw)
    assert raw["invoices"].duplicated().sum() > 0, "the generator must inject duplicate rows"
    for name, key in [("invoices", "invoice_id"), ("shipments", "bl_id"), ("declarations", "declaration_id"), ("payments", "payment_id")]:
        assert clean[name][key].is_unique, name
    assert (raw["invoices"]["hs_code"] != raw["invoices"]["hs_code"].str.strip()).any()
    assert clean["invoices"]["hs_code"].str.fullmatch(r"\d{6}").all()
    assert clean["invoices"]["currency"].str.isupper().all()
    assert set(clean["invoices"]["invoice_id"]) == set(raw["invoices"]["invoice_id"]), "cleaning must not lose trades"


def test_linkage_one_row_per_invoice(dataset, linked) -> None:
    assert linked["trade_id"].is_unique
    assert len(linked) == dataset.tables["invoices"]["invoice_id"].nunique()
    assert linked["has_payment"].eq(1).all()
    assert linked["invoice_date"].is_monotonic_increasing
    # declarations are linked by invoice number first and by bill of lading otherwise
    assert set(linked["customs_link"]) <= {"invoice", "bl", ""}
    assert (linked["customs_link"] == "bl").sum() > 0, "duplicate invoices reach their declaration through the B/L"
    paid = dataset.tables["payments"].drop_duplicates()
    fx = fx_dict(dataset.tables["fx_rates"])
    total = (paid["amount"] * paid["currency"].map(fx)).groupby(paid["invoice_id"]).sum()
    np.testing.assert_allclose(linked.set_index("trade_id")["paid_usd"], total.reindex(linked["trade_id"]), rtol=1e-9)


def test_linking_a_subset_equals_subsetting_the_link(dataset, linked) -> None:
    """An FL client that links only its own records gets the same rows as the pooled linkage."""
    ids = set(linked["trade_id"].sample(frac=0.3, random_state=1))
    t = dataset.tables
    inv = t["invoices"][t["invoices"]["invoice_id"].isin(ids)]
    bls = set(inv["bl_id"].str.strip().str.upper())
    own = {
        "invoices": inv,
        "payments": t["payments"][t["payments"]["invoice_id"].isin(ids)],
        "shipments": t["shipments"][t["shipments"]["bl_id"].isin(bls)],
        "declarations": t["declarations"][t["declarations"]["invoice_id"].isin(ids) | t["declarations"]["bl_id"].isin(bls)],
    }
    local = link_records(own, fx_dict(t["fx_rates"])).set_index("trade_id").sort_index()
    pooled = linked[linked["trade_id"].isin(ids)].set_index("trade_id").sort_index()
    pd.testing.assert_frame_equal(local, pooled, check_dtype=False)


# --------------------------------------------------------------------------- split


def test_split_is_time_ordered(cfg, linked, features) -> None:
    train_end, val_end = split_cutoffs(cfg)
    assert train_end < val_end
    split = assign_split(linked["invoice_date"], cfg)
    dates = linked["invoice_date"]
    assert dates[split == "train"].max() <= train_end < dates[split == "val"].min()
    assert dates[split == "val"].max() <= val_end < dates[split == "test"].min()
    share = pd.Series(split).value_counts(normalize=True)
    assert abs(share["train"] - cfg.split.train_frac) < 0.05
    assert abs(share["val"] - cfg.split.val_frac) < 0.05

    summary = check_stratification(features.split, features.y, features.typology, min_count=0)
    for name in ["train", "val", "test"]:
        assert 0.01 < summary[name]["illicit_rate"] < 0.10
    assert all(count > 0 for count in summary["train"]["typology_counts"].values())
    with pytest.raises(ValueError):
        check_stratification(features.split, features.y, features.typology, min_count=10**6)


# --------------------------------------------------------------------------- indicators


def test_dictionary_is_consistent() -> None:
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == len(FEATURES)
    assert set(ROW_FEATURES) | set(CONTEXT_FEATURES) == set(FEATURE_NAMES)
    assert {f.group for f in FEATURES} == set(GROUPS)
    for spec in FEATURES:
        assert spec.owner in {"bank", "trade", "customs", "cross"}
        assert spec.typologies and set(spec.typologies) <= set(TYPOLOGIES), spec.name
    for typology in TYPOLOGIES:
        assert any(typology in f.typologies for f in FEATURES), f"no indicator targets {typology}"
    text = features_markdown()
    assert all(f"`{name}`" in text for name in FEATURE_NAMES)


def test_feature_matrix_shape_and_values(features, linked) -> None:
    assert isinstance(features, FeatureSet)
    assert features.X.shape == (len(linked), len(FEATURE_NAMES))
    assert features.X.dtype == np.float32
    assert features.feature_names == FEATURE_NAMES
    assert np.isfinite(features.X).all()
    assert set(np.unique(features.y)) == {0, 1}
    assert ((features.typology > 0) == (features.y == 1)).all()
    assert set(features.split) == {"train", "val", "test"}
    flags = [n for n in FEATURE_NAMES if n.startswith(("missing_", "hs_")) or n.endswith(("_flag", "_offshore", "_related"))]
    for name in flags:
        assert set(np.unique(features.X[:, FEATURE_NAMES.index(name)])) <= {0.0, 1.0}, name


def test_save_and_load(features, tmp_path) -> None:
    path = features.save(tmp_path / "f.npz")
    loaded = FeatureSet.load(path)
    np.testing.assert_array_equal(loaded.X, features.X)
    np.testing.assert_array_equal(loaded.y, features.y)
    assert list(loaded.trade_id) == list(features.trade_id)
    assert loaded.feature_names == features.feature_names
    assert len(loaded.part("test")) == int((features.split == "test").sum())


# the indicator that defines each typology must respond to it (one-dimensional separation)
KEY_INDICATOR = {
    "over_invoicing": "price_z",
    "under_invoicing": "price_z",
    "multiple_invoicing": "invoices_per_bl",
    "phantom_shipment": "missing_customs",
    "over_under_shipment": "customs_qty_abs_dev",
    "misdescription": "hs_mismatch",
    "third_party_payment": "third_party_payer",
    "circular_trading": "edge_in_cycle_same_hs",
    "shell_company": "min_party_age_log",
    "high_risk_routing": "route_detour_ratio",
}


@pytest.mark.parametrize("typology", TYPOLOGIES)
def test_key_indicator_responds_to_typology(features, typology) -> None:
    name = KEY_INDICATOR[typology]
    assert typology in FEATURE_BY_NAME[name].typologies, "the dictionary must list the typology for its key indicator"
    k = TYPOLOGY_CLASSES.index(typology)
    rows = (features.typology == 0) | (features.typology == k)
    auc = rank_auc(features.X[rows, FEATURE_NAMES.index(name)], features.typology[rows] == k)
    assert max(auc, 1 - auc) > 0.70, (typology, name, auc)


def test_indicator_directions(features) -> None:
    X = pd.DataFrame(features.X, columns=FEATURE_NAMES)
    typ = features.typology
    legit = typ == 0

    def mean(name: str, typology: str) -> float:
        return float(X.loc[typ == TYPOLOGY_CLASSES.index(typology), name].mean())

    assert mean("price_z", "over_invoicing") > X.loc[legit, "price_z"].mean() + 1.0
    assert mean("price_z", "under_invoicing") < X.loc[legit, "price_z"].mean() - 1.0
    assert mean("invoices_per_bl", "multiple_invoicing") > 1.8
    assert mean("ship_qty_log_ratio", "multiple_invoicing") < -0.4, "each duplicate invoice claims the whole shipment"
    assert abs(X.loc[legit & (X["invoices_per_bl"] > 1), "ship_qty_log_ratio"].median()) < 0.1, "benign split invoices add up to the shipment"
    assert mean("missing_both", "phantom_shipment") > 0.25  # 60% expected; the test dataset has ~30 cases
    assert mean("payer_risk", "third_party_payment") > X.loc[legit, "payer_risk"].mean() + 0.2
    assert mean("cycle_len_same_hs", "circular_trading") >= 2.5
    assert mean("route_detour_ratio", "high_risk_routing") > X.loc[legit, "route_detour_ratio"].mean() + 0.2


# --------------------------------------------------------------------------- leakage guards


def test_labels_do_not_influence_features(dataset, linked, cfg, features) -> None:
    ref = Reference.from_tables(dataset.tables)
    shuffled = dataset.tables["labels"].copy()
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(shuffled))
    shuffled[["is_illicit", "typology"]] = shuffled[["is_illicit", "typology"]].to_numpy()[perm]
    other = build_features(linked, dataset.tables["kyc_entities"], dataset.tables["banks"], ref, cfg, shuffled)
    np.testing.assert_array_equal(other.X, features.X)
    unlabeled = build_features(linked, dataset.tables["kyc_entities"], dataset.tables["banks"], ref, cfg, None)
    np.testing.assert_array_equal(unlabeled.X, features.X)


def test_no_look_ahead_in_context_features(dataset, linked, cfg, features) -> None:
    """Train-period indicators must not change when later trades are removed."""
    ref = Reference.from_tables(dataset.tables)
    train_end, val_end = split_cutoffs(cfg)
    past = linked[linked["invoice_date"] <= train_end]
    only_train = build_features(past, dataset.tables["kyc_entities"], dataset.tables["banks"], ref, cfg, dataset.tables["labels"])
    full_train = features.part("train")
    assert list(only_train.trade_id) == list(full_train.trade_id)
    np.testing.assert_allclose(only_train.X, full_train.X, rtol=1e-6, atol=1e-6)

    upto_val = linked[linked["invoice_date"] <= val_end]
    no_test = build_features(upto_val, dataset.tables["kyc_entities"], dataset.tables["banks"], ref, cfg, dataset.tables["labels"])
    np.testing.assert_allclose(no_test.part("val").X, features.part("val").X, rtol=1e-6, atol=1e-6)


def test_row_features_depend_only_on_the_row(dataset, linked, cfg) -> None:
    ref = Reference.from_tables(dataset.tables)
    kyc, banks = dataset.tables["kyc_entities"], dataset.tables["banks"]
    full = row_features(linked, kyc, banks, ref, cfg)
    sample = linked.sample(n=200, random_state=3)
    part = row_features(sample, kyc, banks, ref, cfg)
    pd.testing.assert_frame_equal(part, full.loc[part.index], check_exact=False, rtol=1e-9)


def test_context_features_use_only_local_records(dataset, linked, cfg, features) -> None:
    """A party that sees fewer trades gets different (thinner) network indicators."""
    ref = Reference.from_tables(dataset.tables)
    half = linked.iloc[::2]
    local = build_features(half, dataset.tables["kyc_entities"], dataset.tables["banks"], ref, cfg, dataset.tables["labels"])
    pooled = features.subset(np.isin(features.trade_id, local.trade_id))
    assert list(local.trade_id) == list(pooled.trade_id)
    row_idx = [FEATURE_NAMES.index(n) for n in ROW_FEATURES]
    np.testing.assert_allclose(local.X[:, row_idx], pooled.X[:, row_idx], rtol=1e-6, atol=1e-6)
    j = FEATURE_NAMES.index("exporter_n_trades_log")
    assert local.X[:, j].mean() < pooled.X[:, j].mean()
    j = FEATURE_NAMES.index("edge_in_cycle")
    assert local.X[:, j].sum() <= pooled.X[:, j].sum()


def test_feature_report(features, cfg) -> None:
    report = feature_report(features, cfg)
    assert report["n_features"] == len(FEATURE_NAMES)
    assert set(report["typologies"]) == set(TYPOLOGIES)
    for typology, info in report["typologies"].items():
        assert info["top_features"][0]["separation"] > 0.7, typology
