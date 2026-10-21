"""Run with:  pytest -q
Uses a small fake dataset in the IBM format, so it works without the real download."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from aml_fl import data, features, partition, patterns, profiles, split  # noqa: E402
from aml_fl.metrics import recall_by_pattern  # noqa: E402
from aml_fl.preprocess import TabularPreprocessor, undersample  # noqa: E402
from aml_fl.utils import load_config  # noqa: E402
from make_toy_data import make_toy, make_toy_with_patterns, write_ibm_csv, write_patterns_txt  # noqa: E402


@pytest.fixture(scope="module")
def cleaned(tmp_path_factory):
    path = tmp_path_factory.mktemp("raw") / "toy.csv"
    write_ibm_csv(make_toy(rows=5000, seed=1), path)
    df, rates = data.clean(data.load_raw(path))
    return df, rates


def test_load_and_clean(cleaned):
    df, rates = cleaned
    assert df["timestamp"].is_monotonic_increasing
    assert {"src_key", "dst_key", "amount_usd", "txn_id"} <= set(df.columns)
    assert rates["US Dollar"] == 1.0
    assert abs(rates["Rupee"] - 74.0) < 1.0  # recovered from the data itself


def test_features_use_past_only(cleaned):
    df, _ = cleaned
    f = features.build_features(df)
    assert set(features.TXN_FEATURES) <= set(f.columns)
    first = f.groupby("src_key").head(1)
    # The first transaction of every account has no history.
    assert (first["src_txn_count"] == 0).all()
    assert (first["src_fan_out"] == 0).all()
    assert (first["src_burst"] == 0).all()
    assert f[features.NUMERIC].notna().all().all()
    # Changing a FUTURE transaction must not change features of earlier rows.
    df2 = df.copy()
    df2.loc[df2.index[-1], "amount_usd"] *= 1000
    f2 = features.build_features(df2)
    pd.testing.assert_frame_equal(f[features.NUMERIC].iloc[:-1], f2[features.NUMERIC].iloc[:-1])


def test_time_split_is_chronological(cleaned):
    df, _ = cleaned
    s = split.time_split(df, 0.6, 0.2)
    assert df.loc[s == "train", "timestamp"].max() < df.loc[s == "val", "timestamp"].min()
    assert df.loc[s == "val", "timestamp"].max() < df.loc[s == "test", "timestamp"].min()


def test_profiles_signal_zero_is_independent_of_labels():
    accts = pd.Series([f"a{i}" for i in range(4000)])
    risky = pd.Series(np.arange(4000) < 2000)
    p0 = profiles.generate_profiles(accts, risky, label_signal=0.0, seed=0)
    p1 = profiles.generate_profiles(accts, risky, label_signal=0.5, seed=0)
    young0 = (p0["account_age_days"] < 180).groupby(risky.values).mean()
    young1 = (p1["account_age_days"] < 180).groupby(risky.values).mean()
    assert abs(young0[True] - young0[False]) < 0.05   # no signal
    assert young1[True] - young1[False] > 0.2          # signal present


def test_partition_gives_different_schemas(cleaned):
    df, _ = cleaned
    cfg = load_config(ROOT / "configs" / "default.yaml")
    f = features.build_features(df)
    f["split"] = split.time_split(f)
    f = profiles.attach_profiles(f, f["split"], 0.1, seed=0)
    clients, bank_map = partition.partition(f, 5, "group", cfg["schemas"])
    assert sum(len(c) for c in clients.values()) == len(f)          # nothing lost
    schemas = [tuple(c.columns) for c in clients.values()]
    assert len(set(schemas)) == 5                                      # all different
    ids = [set(c["txn_id"]) for c in clients.values()]
    assert all(ids[i].isdisjoint(ids[j]) for i in range(5) for j in range(i + 1, 5))


def test_preprocessor_and_undersample(cleaned):
    df, _ = cleaned
    f = features.build_features(df)
    f["split"] = split.time_split(f)
    sub = f[partition.ID_COLUMNS + ["log_amount", "payment_format", "src_fan_out"]]
    pre = TabularPreprocessor.for_frame(sub).fit(sub[sub["split"] == "train"])
    X = pre.transform(sub)
    assert X.shape == (len(sub), len(pre.feature_names))
    assert np.isfinite(X).all()
    y = sub["label"].to_numpy()
    Xs, ys = undersample(X, y, neg_per_pos=5)
    assert ys.sum() == y.sum() and len(ys) <= y.sum() * 6


def test_patterns_parse_and_tag(tmp_path):
    raw, attempts = make_toy_with_patterns(rows=5000, seed=1)
    write_ibm_csv(raw, tmp_path / "toy_Trans.csv")
    write_patterns_txt(raw, attempts, tmp_path / "toy_Patterns.txt")
    assert patterns.default_path(tmp_path / "toy_Trans.csv") == tmp_path / "toy_Patterns.txt"

    parsed = patterns.parse_patterns(tmp_path / "toy_Patterns.txt")
    assert parsed["attempt_id"].nunique() == len(attempts)
    assert set(parsed["pattern"]) == {"fan_in", "fan_out", "cycle", "scatter_gather"}

    df, _ = data.clean(data.load_raw(tmp_path / "toy_Trans.csv"))
    tags = patterns.tag_transactions(df, parsed)
    # Every laundering row was injected by a listed attempt, so none is "unlisted".
    assert (tags[df["label"] == 1] != patterns.UNLISTED).all()
    assert (tags[df["label"] == 0] == patterns.NONE).all()
    assert tags.value_counts().drop(patterns.NONE).to_dict() == parsed["pattern"].value_counts().to_dict()


def test_recall_by_pattern():
    pats = np.array(["none", "fan_in", "fan_in", "cycle", "cycle", "cycle"])
    scores = np.array([0.9, 0.8, 0.1, 0.7, 0.2, 0.1])
    out = recall_by_pattern(pats, scores, threshold=0.5).set_index("pattern")
    assert out.loc["fan_in", "recall"] == 0.5
    assert out.loc["cycle", "positives"] == 3 and out.loc["cycle", "detected"] == 1
    assert "none" not in out.index


def test_model_shared_weights_roundtrip():
    torch = pytest.importorskip("torch")
    from aml_fl.models import BankModel

    a, b = BankModel(input_dim=14), BankModel(input_dim=9)   # different schemas
    b.set_shared_weights(a.get_shared_weights())               # heads are compatible
    z = torch.randn(4, 32)
    assert torch.allclose(a.head(z), b.head(z))
    assert a(torch.randn(3, 14)).shape == (3,)
