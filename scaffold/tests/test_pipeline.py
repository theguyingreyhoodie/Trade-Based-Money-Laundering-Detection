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


def test_trim_drops_rows_at_or_after_end_time(cleaned):
    df, _ = cleaned
    end = "2022-09-05 00:00"
    t = data.trim(df, end)
    assert (t["timestamp"] < pd.Timestamp(end)).all()
    assert len(t) == int((df["timestamp"] < pd.Timestamp(end)).sum())
    assert data.trim(df, None) is df


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
    core = cfg["core_features"]
    clients, bank_map = partition.partition(f, 5, "group", cfg["schemas"], core_features=core)
    assert sum(len(c) for c in clients.values()) == len(f)          # nothing lost
    schemas = [tuple(c.columns) for c in clients.values()]
    assert len(set(schemas)) == 5                                      # all different
    assert all(set(core) <= set(s) for s in schemas)                   # core shared by all
    assert all(len(s) == len(set(s)) for s in schemas)                 # no duplicate columns
    with pytest.raises(ValueError):
        partition.partition(f, 5, "group", cfg["schemas"], core_features=["not_a_feature"])
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


def test_pooled_evaluate_uses_each_banks_threshold():
    from aml_fl.metrics import pooled_evaluate

    y = np.array([1, 0, 1, 0])
    scores = np.array([0.3, 0.2, 0.9, 0.8])
    thresholds = np.array([0.25, 0.25, 0.85, 0.85])      # bank A: rows 0-1, bank B: rows 2-3
    out = pooled_evaluate(y, scores, thresholds)
    assert out["recall"] == 1.0 and out["fpr"] == 0.0   # 0.3>=0.25 and 0.9>=0.85 flagged; 0.2, 0.8 not
    assert np.isnan(out["threshold"])


def test_recall_by_pattern():
    pats = np.array(["none", "fan_in", "fan_in", "cycle", "cycle", "cycle"])
    scores = np.array([0.9, 0.8, 0.1, 0.7, 0.2, 0.1])
    out = recall_by_pattern(pats, scores, threshold=0.5).set_index("pattern")
    assert out.loc["fan_in", "recall"] == 0.5
    assert out.loc["cycle", "positives"] == 3 and out.loc["cycle", "detected"] == 1
    assert "none" not in out.index


def test_long_tailed_columns_are_logged_before_scaling():
    df = pd.DataFrame({"src_txn_count": [0, 1, 10, 100_000], "hour": [0, 1, 10, 23]})
    pre = TabularPreprocessor.for_frame(df).fit(df)
    X = pre.transform(df)
    logged = np.log1p(df["src_txn_count"])
    expected = (logged - logged.mean()) / logged.std()
    np.testing.assert_allclose(X[:, pre.numeric.index("src_txn_count")], expected, rtol=1e-5)
    hour = (df["hour"] - df["hour"].mean()) / df["hour"].std()          # not in LOG1P_COLUMNS
    np.testing.assert_allclose(X[:, pre.numeric.index("hour")], hour, rtol=1e-5)


def test_early_stopping_keeps_best_epoch():
    torch = pytest.importorskip("torch")
    from aml_fl.models import BankModel, predict_proba, train_early_stopping
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5)).astype(np.float32)
    y = (X[:, 0] + 0.3 * rng.normal(size=2000) > 1.5).astype(np.float32)
    torch.manual_seed(0)
    model = BankModel(input_dim=5)
    best, epoch = train_early_stopping(model, X[:1500], y[:1500], X[1500:], y[1500:], max_epochs=30,
                                       patience=10, lr=1e-2, batch_size=128)
    assert 1 <= epoch <= 30 and best > 0.8
    assert abs(average_precision_score(y[1500:], predict_proba(model, X[1500:])) - best) < 1e-6  # best weights kept


def test_shared_vocab_aligns_one_hot_layout():
    a = pd.DataFrame({"payment_format": ["Wire", "Cash"], "log_amount": [1.0, 2.0]})
    b = pd.DataFrame({"payment_format": ["ACH", "Bitcoin"], "log_amount": [3.0, 4.0]})
    vocab = {"payment_format": features.PAYMENT_FORMATS}
    pa = TabularPreprocessor.for_frame(a, vocab).fit(a)
    pb = TabularPreprocessor.for_frame(b, vocab).fit(b)
    assert pa.feature_names == pb.feature_names           # same columns, same order
    assert pa.transform(a).shape[1] == len(features.PAYMENT_FORMATS) + 2   # + <other> + log_amount


def test_fl_core_input_dim_matches_client(tmp_path):
    from aml_fl.preprocess import load_client
    from aml_fl.utils import save_json, save_table
    from run_fl import core_input_dim

    cfg = load_config(ROOT / "configs" / "default.yaml")
    write_ibm_csv(make_toy(rows=3000, seed=2), tmp_path / "toy.csv")
    df, _ = data.clean(data.load_raw(tmp_path / "toy.csv"))
    f = features.build_features(df)
    f["split"] = split.time_split(f)
    core = cfg["core_features"]
    save_table(f[partition.ID_COLUMNS + core], tmp_path / "client_0" / "transactions")
    save_json({"columns": core, "core_columns": core}, tmp_path / "client_0" / "schema.json")
    c = load_client(tmp_path / "client_0", "core", features.SHARED_VOCAB)
    assert c.input_dim == core_input_dim(core)


def test_audit_hash_is_deterministic_and_sensitive():
    pytest.importorskip("flwr")
    from flwr.app import ArrayRecord

    from aml_fl.fl.strategy import hash_parameters

    w = [np.arange(6, dtype=np.float32).reshape(2, 3), np.zeros(3, dtype=np.float32)]
    h1 = hash_parameters(ArrayRecord(numpy_ndarrays=w))
    assert h1 == hash_parameters(ArrayRecord(numpy_ndarrays=[x.copy() for x in w]))
    w[1][0] = 1e-6
    assert h1 != hash_parameters(ArrayRecord(numpy_ndarrays=w))
    assert len(h1) == 64


@pytest.fixture(scope="module")
def toy_partitions(tmp_path_factory, cleaned):
    """Client folders as prepare_data.py writes them, from the toy data."""
    from aml_fl.utils import save_json, save_table

    df, _ = cleaned
    cfg = load_config(ROOT / "configs" / "default.yaml")
    f = features.build_features(df)
    f["split"] = split.time_split(f)
    f = profiles.attach_profiles(f, f["split"], 0.1, seed=0)
    core = cfg["core_features"]
    clients, _ = partition.partition(f, 5, "group", cfg["schemas"], core_features=core)
    out = tmp_path_factory.mktemp("partitions")
    for name, cdf in clients.items():
        save_table(cdf, out / name / "transactions")
        save_json({"client": name, "columns": [c for c in cdf.columns if c not in partition.ID_COLUMNS],
                   "core_columns": core}, out / name / "schema.json")
    return out


def _train_message(arrays, rnd: int):
    import time

    from flwr.app import ConfigRecord, Message, Metadata, RecordDict

    md = Metadata(run_id=1, message_id=f"m{rnd}", src_node_id=0, dst_node_id=7, reply_to_message_id="",
                  group_id="", created_at=time.time(), ttl=3600, message_type="train")
    return Message(content=RecordDict({"arrays": arrays, "config": ConfigRecord({"server-round": rnd})}),
                   metadata=md)


def test_fl_encoder_head_only_shares_the_head(toy_partitions, tmp_path):
    pytest.importorskip("flwr")
    from flwr.app import ArrayRecord, Context, RecordDict

    from aml_fl.fl.client_app import ENCODER_KEY, make_client_app
    from aml_fl.models import SharedHead

    app = make_client_app(toy_partitions, tmp_path, mode="encoder_head")
    ctx = Context(run_id=1, node_id=7, node_config={"partition-id": 1}, state=RecordDict(), run_config={})
    head = {f"head.{k}": v for k, v in SharedHead().state_dict().items()}
    head_keys = set(head)
    for rnd in (1, 2):
        reply = app(_train_message(ArrayRecord(head), rnd), ctx)
        assert set(reply.content["arrays"].keys()) == head_keys        # nothing but the head leaves
        assert reply.content["metrics"]["encoder_rounds"] == rnd        # encoder kept between rounds
        assert ENCODER_KEY in ctx.state                                  # ... at the bank only


def test_fl_hybrid_shares_core_encoder_and_head(toy_partitions, tmp_path):
    pytest.importorskip("flwr")
    from flwr.app import ArrayRecord, Context, RecordDict

    from aml_fl.fl.client_app import ENCODER_KEY, make_client_app, shared_state
    from aml_fl.models import HybridModel
    from run_fl import core_input_dim

    cfg = load_config(ROOT / "configs" / "default.yaml")
    initial = shared_state(HybridModel(core_input_dim(cfg["core_features"]), 1), "hybrid")
    assert initial and all(k.startswith(("core_encoder.", "head.")) for k in initial)
    app = make_client_app(toy_partitions, tmp_path, mode="hybrid")
    for pid in (0, 3):  # banks with different own schemas accept the same global weights
        ctx = Context(run_id=1, node_id=7, node_config={"partition-id": pid}, state=RecordDict(), run_config={})
        for rnd in (1, 2):
            reply = app(_train_message(ArrayRecord(initial), rnd), ctx)
            assert set(reply.content["arrays"].keys()) == set(initial)   # private encoder stays home
            assert reply.content["metrics"]["encoder_rounds"] == rnd
        assert all(k.startswith("private_encoder.") for k in ctx.state[ENCODER_KEY].keys())


def test_hybrid_loader_puts_shared_core_first(toy_partitions):
    from aml_fl.preprocess import load_client_hybrid
    from run_fl import core_input_dim

    cfg = load_config(ROOT / "configs" / "default.yaml")
    a = load_client_hybrid(toy_partitions / "client_0", features.SHARED_VOCAB)
    b = load_client_hybrid(toy_partitions / "client_2", features.SHARED_VOCAB)
    assert a.core_dim == b.core_dim == core_input_dim(cfg["core_features"])
    assert a.feature_names[:a.core_dim] == b.feature_names[:b.core_dim]
    assert a.feature_names[a.core_dim:] != b.feature_names[b.core_dim:]   # own layouts differ


def test_zero_filled_layout_is_shared_and_flags_missing(toy_partitions):
    from aml_fl.fl.client_app import ALL_VOCAB
    from aml_fl.preprocess import load_client_zero_filled
    from aml_fl.utils import load_json
    from run_fl import agreed_columns

    columns = agreed_columns(toy_partitions)
    a = load_client_zero_filled(toy_partitions / "client_0", columns, ALL_VOCAB)
    b = load_client_zero_filled(toy_partitions / "client_2", columns, ALL_VOCAB)
    assert a.feature_names == b.feature_names                       # one layout for every bank
    own = load_json(toy_partitions / "client_0" / "schema.json")["columns"]
    for c in columns:
        flag = a.X["train"][:, a.feature_names.index(f"{c}=<missing>")] if f"{c}=<missing>" in a.feature_names else None
        cols = [i for i, n in enumerate(a.feature_names) if n == c or n.startswith(f"{c}=") and not n.endswith("<missing>")]
        if c in own:
            assert flag is None or (flag == 0).all()
        else:                                                       # lacked: zeros + flag
            assert (flag == 1).all() and (a.X["train"][:, cols] == 0).all()


def test_fl_zero_filled_shares_the_whole_model(toy_partitions, tmp_path):
    pytest.importorskip("flwr")
    from flwr.app import ArrayRecord, Context, RecordDict

    from aml_fl.fl.client_app import ALL_VOCAB, make_client_app
    from aml_fl.models import BankModel
    from aml_fl.preprocess import zero_filled_layout
    from run_fl import agreed_columns

    cfg = load_config(ROOT / "configs" / "default.yaml")
    columns = agreed_columns(toy_partitions)
    initial = BankModel(input_dim=len(zero_filled_layout(columns, ALL_VOCAB, cfg["core_features"]))).state_dict()
    app = make_client_app(toy_partitions, tmp_path, mode="zero_filled", global_columns=columns)
    for pid in (1, 4):
        ctx = Context(run_id=1, node_id=7, node_config={"partition-id": pid}, state=RecordDict(), run_config={})
        reply = app(_train_message(ArrayRecord(initial), 1), ctx)
        assert set(reply.content["arrays"].keys()) == set(initial)
        assert len(ctx.state) == 0                                  # nothing kept private


def test_fl_naive_shares_the_whole_model(toy_partitions, tmp_path):
    pytest.importorskip("flwr")
    from flwr.app import ArrayRecord, Context, RecordDict

    from aml_fl.fl.client_app import make_client_app
    from aml_fl.models import BankModel
    from run_fl import core_input_dim

    cfg = load_config(ROOT / "configs" / "default.yaml")
    initial = BankModel(input_dim=core_input_dim(cfg["core_features"])).state_dict()
    app = make_client_app(toy_partitions, tmp_path, mode="naive")
    ctx = Context(run_id=1, node_id=7, node_config={"partition-id": 0}, state=RecordDict(), run_config={})
    reply = app(_train_message(ArrayRecord(initial), 1), ctx)
    assert set(reply.content["arrays"].keys()) == set(initial)


def test_model_shared_weights_roundtrip():
    torch = pytest.importorskip("torch")
    from aml_fl.models import BankModel

    a, b = BankModel(input_dim=14), BankModel(input_dim=9)   # different schemas
    b.set_shared_weights(a.get_shared_weights())               # heads are compatible
    z = torch.randn(4, 32)
    assert torch.allclose(a.head(z), b.head(z))
    assert a(torch.randn(3, 14)).shape == (3,)
