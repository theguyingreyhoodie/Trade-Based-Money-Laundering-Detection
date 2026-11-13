"""Federated learning: partitioning, client data, the client boundary, and a simulated run."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord, ConfigRecord, MetricRecord, RecordDict

from tbml.data.typologies import TYPOLOGY_CLASSES
from tbml.eval.metrics import N_CLASSES, ScoreHist
from tbml.features.dictionary import FEATURE_NAMES, ROW_FEATURES
from tbml.fl.boundary import EVALUATE, STATS, TRAIN, BoundarySpec, BoundaryViolation, check_reply, message_bytes
from tbml.fl.client import local_evaluation, local_update
from tbml.fl.data import load_client, prepare_clients, scenario_dir
from tbml.fl.experiments import experiment_plan, primary_scenario
from tbml.fl.local import run_local_only, run_pooled_local_features
from tbml.fl.partition import Scenario, partition_summary, partition_trades
from tbml.fl.simulation import load_global_model, run_federated
from tbml.ledger.hashing import hash_arrays
from tbml.models.mlp import feature_moments, get_weights, make_model, moments_to_scaler, predict_proba, weight_spec

# --------------------------------------------------------------------------- partitioning


def _typology(n: int = 30000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.zeros(n, dtype=np.int64)
    illicit = rng.random(n) < 0.04
    t[illicit] = rng.integers(1, N_CLASSES, illicit.sum())
    return t


def _mix_skew(owner: np.ndarray, typology: np.ndarray) -> float:
    """Mean total-variation distance between each client's typology mix and the overall mix."""
    illicit = typology > 0
    overall = np.bincount(typology[illicit], minlength=N_CLASSES)[1:] / illicit.sum()
    dists = []
    for c in range(owner.max() + 1):
        m = illicit & (owner == c)
        mix = np.bincount(typology[m], minlength=N_CLASSES)[1:] / max(m.sum(), 1)
        dists.append(0.5 * np.abs(mix - overall).sum())
    return float(np.mean(dists))


def test_scenario_names_and_validation() -> None:
    assert Scenario(3, "iid").name == "n3_iid"
    assert Scenario(5, "dirichlet", 0.5).name == "n5_dir0.5"
    assert Scenario(3, "dirichlet", 1.0).name == "n3_dir1"
    assert Scenario(10, "quantity", 0.5).name == "n10_qty0.5"
    assert Scenario(3, "iid").is_iid and not Scenario(3, "dirichlet", 0.1).is_iid
    for bad in [dict(n_clients=1), dict(n_clients=3, partition="weird"), dict(n_clients=3, partition="dirichlet"), dict(n_clients=3, partition="quantity", alpha=0.0)]:
        with pytest.raises(ValueError):
            Scenario(**bad)


def test_iid_partition_is_balanced() -> None:
    t = _typology()
    owner = partition_trades(t, Scenario(3, "iid"), seed=1)
    sizes = np.bincount(owner)
    assert len(sizes) == 3 and sizes.max() - sizes.min() <= 1
    rates = [float((t[owner == c] > 0).mean()) for c in range(3)]
    assert max(rates) - min(rates) < 0.01
    assert _mix_skew(owner, t) < 0.12
    np.testing.assert_array_equal(owner, partition_trades(t, Scenario(3, "iid"), seed=1))
    assert not np.array_equal(owner, partition_trades(t, Scenario(3, "iid"), seed=2))


def test_dirichlet_skew_grows_as_alpha_shrinks() -> None:
    t = _typology()
    skew = {a: np.mean([_mix_skew(partition_trades(t, Scenario(3, "dirichlet", a), seed=s), t) for s in range(8)]) for a in (0.1, 0.5, 1.0, 100.0)}
    assert skew[0.1] > skew[0.5] > skew[1.0] > skew[100.0]
    assert skew[0.1] > 0.4, "alpha = 0.1 must give strongly different typology mixes"
    # label skew: the positive rate differs between clients
    owner = partition_trades(t, Scenario(3, "dirichlet", 0.1), seed=3)
    summary = partition_summary(owner, t)
    low, high = summary["illicit_rate_range"]
    assert high > 1.5 * low
    assert sum(c["n"] for c in summary["clients"]) == len(t)
    assert set(summary["clients"][0]["typology_counts"]) == set(TYPOLOGY_CLASSES[1:])


@pytest.mark.parametrize("scenario", [Scenario(3, "dirichlet", 0.1), Scenario(3, "dirichlet", 0.5), Scenario(5, "dirichlet", 0.5), Scenario(10, "dirichlet", 0.5), Scenario(3, "quantity", 0.5)])
def test_no_degenerate_clients(scenario: Scenario) -> None:
    """Every client keeps at least a quarter of an equal share of the legitimate trades."""
    t = _typology(60000)
    for seed in range(5):
        owner = partition_trades(t, scenario, seed=seed, min_share=0.25)
        assert set(np.unique(owner)) == set(range(scenario.n_clients)), "every trade is assigned to exactly one existing client"
        legit = np.bincount(owner[t == 0], minlength=scenario.n_clients)
        floor = 0.25 * (t == 0).sum() / scenario.n_clients
        if scenario.partition == "dirichlet":
            assert legit.min() >= floor * 0.98, (scenario.name, seed, legit)
        rates = np.array([(t[owner == c] > 0).mean() for c in range(scenario.n_clients)])
        assert rates.max() < 0.6, "no client may consist mostly of illicit trades"


def test_quantity_skew_changes_sizes_not_labels() -> None:
    t = _typology(60000)
    owner = partition_trades(t, Scenario(3, "quantity", 0.5), seed=4)
    sizes = np.bincount(owner, minlength=3)
    assert sizes.max() / sizes.min() > 1.3
    rates = [float((t[owner == c] > 0).mean()) for c in range(3)]
    assert max(rates) - min(rates) < 0.015


def test_experiment_plan(cfg) -> None:
    from tbml.config import load_config

    full = load_config()
    plan = experiment_plan(full)
    names = [s.name for s, _ in plan]
    assert len(names) == len(set(names))
    assert {"n3_iid", "n3_dir0.1", "n3_dir0.5", "n3_dir1", "n3_qty0.5", "n5_iid", "n10_iid", "n5_dir0.5", "n10_dir0.5"} == set(names)
    for scenario, strategies in plan:
        assert "fedavg" in strategies
        if scenario.n_clients == 3 and not scenario.is_iid:
            assert "fedprox" in strategies, "FedProx is the non-IID comparison point"
    assert all(s.n_clients >= 3 for s, _ in plan), "the proposal requires at least 3 data owners"
    assert primary_scenario(full).n_clients >= 3


# --------------------------------------------------------------------------- client data


@pytest.fixture(scope="module")
def scenario() -> Scenario:
    return Scenario(3, "dirichlet", 0.5)


@pytest.fixture(scope="module")
def clients(cfg, dataset, scenario):
    from tbml.data.generator import write_dataset

    write_dataset(dataset, cfg.seed_dir("data_dir", 0))
    summary = prepare_clients(cfg, 0, scenario, force=True)
    root = scenario_dir(cfg, 0, scenario)
    return summary, [load_client(root, c) for c in range(scenario.n_clients)]


def test_clients_partition_the_trades(clients, features) -> None:
    summary, parts = clients
    ids = [set(p.trade_id) for p in parts]
    assert sum(len(i) for i in ids) == len(features), "client data sets are disjoint"
    assert set().union(*ids) == set(features.trade_id), "and together cover every trade"
    assert [c["n"] for c in summary["clients"]] == [len(p) for p in parts]
    for p in parts:
        assert p.feature_names == FEATURE_NAMES and np.isfinite(p.X).all()
        assert set(p.split) <= {"train", "val", "test"}


def test_client_features_come_from_the_local_view(clients, features) -> None:
    """Row indicators equal the pooled ones; context indicators are thinner on a client."""
    _, parts = clients
    pooled = {tid: i for i, tid in enumerate(features.trade_id)}
    rows = [FEATURE_NAMES.index(n) for n in ROW_FEATURES]
    j_trades = FEATURE_NAMES.index("exporter_n_trades_log")
    for p in parts:
        idx = np.array([pooled[t] for t in p.trade_id])
        np.testing.assert_allclose(p.X[:, rows], features.X[idx][:, rows], rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(p.y, features.y[idx])
        assert (p.X[:, j_trades] <= features.X[idx][:, j_trades] + 1e-6).all(), "a client cannot see more trades of an entity than exist"
    assert any((p.X[:, j_trades] < features.X[np.array([pooled[t] for t in p.trade_id])][:, j_trades] - 1e-6).any() for p in parts)


def test_local_update_and_evaluation(cfg, clients) -> None:
    _, parts = clients
    fs = parts[0]
    model = make_model(cfg, fs.X.shape[1], seed=0)
    start = get_weights(model)
    new, metrics = local_update(cfg, fs, start, partition_id=0, server_round=1, seed=0, lr=0.05, local_epochs=1)
    assert metrics["num-examples"] == len(fs.part("train")) and metrics["steps"] > 0
    assert [w.shape for w in new] == [w.shape for w in start]
    assert any(not np.array_equal(a, b) for a, b in zip(new[2:], start[2:])), "training must change the weights"
    np.testing.assert_array_equal(new[0], start[0])  # the shared scaler is not trained
    again, _ = local_update(cfg, fs, start, partition_id=0, server_round=1, seed=0, lr=0.05, local_epochs=1)
    for a, b in zip(new, again):
        np.testing.assert_array_equal(a, b)

    ev = local_evaluation(cfg, fs, new, n_bins=int(cfg.fl.hist_bins))
    assert ScoreHist.from_list(ev["hist-val"]).n == len(fs.part("val"))
    assert ScoreHist.from_list(ev["hist-test"]).n == len(fs.part("test"))
    assert len(ev["hist-val"]) == N_CLASSES * int(cfg.fl.hist_bins)


# --------------------------------------------------------------------------- the client boundary


def _spec(cfg) -> tuple[BoundarySpec, list[str], list[np.ndarray]]:
    model = make_model(cfg, len(FEATURE_NAMES), seed=0)
    spec = BoundarySpec(arrays=weight_spec(model), n_features=len(FEATURE_NAMES), hist_len=N_CLASSES * int(cfg.fl.hist_bins))
    return spec, [n for n, _, _ in weight_spec(model)], get_weights(model)


def _record(names, weights) -> ArrayRecord:
    return ArrayRecord(array_dict=OrderedDict((n, Array(np.asarray(w))) for n, w in zip(names, weights)))


def _train_reply(names, weights, **metrics) -> RecordDict:
    base = {"num-examples": 100, "partition-id": 0, "train-loss": 0.3, "steps": 5}
    base.update(metrics)
    digest = hash_arrays(zip(names, weights))
    return RecordDict({"arrays": _record(names, weights), "metrics": MetricRecord(base), "meta": ConfigRecord({"update-sha256": digest})})


def test_boundary_accepts_legitimate_replies(cfg) -> None:
    spec, names, weights = _spec(cfg)
    receipt = check_reply(_train_reply(names, weights), spec, TRAIN)
    assert receipt["kind"] == TRAIN and receipt["n_arrays"] == len(names)
    assert receipt["bytes"] <= spec.byte_budget(TRAIN)
    hist = [0] * spec.hist_len
    ok_eval = RecordDict({"metrics": MetricRecord({"num-examples": 50, "partition-id": 1, "val-loss": 0.2, "hist-val": hist, "hist-test": hist})})
    assert check_reply(ok_eval, spec, EVALUATE)["n_arrays"] == 0
    ok_stats = RecordDict({"metrics": MetricRecord({"num-examples": 50, "partition-id": 1, "feat-sum": [0.0] * spec.n_features, "feat-sumsq": [0.0] * spec.n_features})})
    assert check_reply(ok_stats, spec, STATS)["metric_keys"] == ["feat-sum", "feat-sumsq", "num-examples", "partition-id"]
    assert len(spec.received) == 3


def test_boundary_rejects_raw_data_in_any_disguise(cfg, features) -> None:
    """A reply cannot carry rows: not as an extra record, a longer list, a bigger array or text."""
    spec, names, weights = _spec(cfg)
    raw = features.X[:500]

    extra_record = _train_reply(names, weights)
    extra_record["raw"] = ArrayRecord(array_dict={"rows": Array(raw)})
    extra_array = RecordDict({"arrays": _record([*names, "rows"], [*weights, raw]), "metrics": MetricRecord({"num-examples": 1, "partition-id": 0}),
                              "meta": ConfigRecord({"update-sha256": "0" * 64})})
    reshaped = [w.copy() for w in weights]
    reshaped[2] = np.zeros((reshaped[2].shape[0] + 50, reshaped[2].shape[1]), dtype=np.float32)
    wrong_dtype = [w.astype(np.float64) for w in weights]
    in_metric = _train_reply(names, weights)
    in_metric["metrics"] = MetricRecord({"num-examples": 1, "partition-id": 0, "rows": [float(v) for v in raw[0]]})
    long_list = RecordDict({"metrics": MetricRecord({"num-examples": 1, "partition-id": 0, "val-loss": 0.1, "hist-val": [1] * (spec.hist_len + 1), "hist-test": [0] * spec.hist_len})})
    scalar_as_list = _train_reply(names, weights, **{"train-loss": [0.1, 0.2, 0.3]})
    in_config = _train_reply(names, weights)
    in_config["meta"] = ConfigRecord({"update-sha256": "0" * 64, "note": "INV-0000001,ENT-00042,125000.00"})
    long_hash = _train_reply(names, weights)
    long_hash["meta"] = ConfigRecord({"update-sha256": "x" * 5000})
    arrays_in_eval = RecordDict({"arrays": _record(names, weights), "metrics": MetricRecord({"num-examples": 1, "partition-id": 0, "val-loss": 0.1})})
    two_metric_records = _train_reply(names, weights)
    two_metric_records["more"] = MetricRecord({"num-examples": 1})

    cases = [
        ("extra array record", extra_record, TRAIN),
        ("extra array inside the model record", extra_array, TRAIN),
        ("array larger than the template", _train_reply(names, reshaped), TRAIN),
        ("array of another dtype", _train_reply(names, wrong_dtype), TRAIN),
        ("unknown metric key", in_metric, TRAIN),
        ("histogram longer than agreed", long_list, EVALUATE),
        ("list where a scalar is expected", scalar_as_list, TRAIN),
        ("free text in the config record", in_config, TRAIN),
        ("over-long hash string", long_hash, TRAIN),
        ("arrays in an evaluation reply", arrays_in_eval, EVALUATE),
        ("arrays in a statistics reply", arrays_in_eval, STATS),
        ("second metric record", two_metric_records, TRAIN),
    ]
    for label, content, kind in cases:
        with pytest.raises(BoundaryViolation):
            check_reply(content, spec, kind)
    assert not spec.received, "rejected replies must not be recorded as received"


def test_reply_size_does_not_depend_on_client_data(cfg, clients) -> None:
    """A big client and a small client send the same number of bytes."""
    _, parts = clients
    spec, names, _ = _spec(cfg)
    sizes = []
    for fs in sorted(parts, key=len)[:: len(parts) - 1]:
        model = make_model(cfg, fs.X.shape[1], seed=0)
        new, metrics = local_update(cfg, fs, get_weights(model), 0, 1, 0, lr=0.05, local_epochs=1)
        sizes.append(check_reply(_train_reply(names, new, **metrics), spec, TRAIN)["bytes"])
    assert abs(sizes[0] - sizes[1]) <= 8, sizes
    assert message_bytes(_train_reply(names, get_weights(make_model(cfg, len(FEATURE_NAMES))))) < spec.model_bytes() * 1.1


def test_update_hash_is_sensitive(cfg) -> None:
    _, names, weights = _spec(cfg)
    base = hash_arrays(zip(names, weights))
    assert base == hash_arrays(zip(names, [w.copy() for w in weights]))
    changed = [w.copy() for w in weights]
    changed[3].flat[0] += 1e-6
    assert hash_arrays(zip(names, changed)) != base
    assert hash_arrays(zip(reversed(names), reversed(weights))) != base
    assert hash_arrays(zip([n + "x" for n in names], weights)) != base


# --------------------------------------------------------------------------- simulated federation


@pytest.fixture(scope="module")
def fed_run(cfg, clients, scenario):
    """One short FedAvg run on the Flower simulation runtime (shared by the tests below)."""
    return run_federated(cfg, 0, scenario, "fedavg", rounds=2, tag="test")


@pytest.mark.slow
def test_federated_run_produces_results(fed_run, scenario) -> None:
    r = fed_run
    assert r["n_clients"] == 3 and r["rounds"] == 2 and len(r["history"]) == 2
    assert 1 <= r["best_round"] <= 2
    for key in ["precision", "recall", "f1", "pr_auc", "roc_auc", "fpr"]:
        assert 0.0 <= r["final"][key] <= 1.0
        assert 0.0 <= r["final_client_thresholds"][key] <= 1.0
    assert r["final"]["pr_auc"] > 3 * r["final"]["prevalence"], "two rounds must already rank far better than chance"
    assert r["history"][1]["train_loss"] < r["history"][0]["train_loss"]
    assert len(r["per_client"]) == 3
    assert sum(c["n_test"] for c in r["per_client"]) == r["final"]["n"]


@pytest.mark.slow
def test_server_only_ever_receives_parameters(fed_run) -> None:
    """The server-side record of everything it received: parameters and whitelisted aggregates only."""
    b = fed_run["boundary"]
    assert b["violations"] == 0
    assert b["kinds"] == {"stats": 3, "train": 6, "evaluate": 6}, "3 clients: one statistics reply, then train + evaluate per round"
    assert b["replies_checked"] == 15
    assert b["array_records_outside_training"] == 0
    model_bytes = fed_run["communication"]["model_bytes"]
    assert b["max_reply_bytes"]["train"] < model_bytes * 1.1 + 4096, "a training reply is the size of the model, not of the data"
    assert b["max_reply_bytes"]["stats"] < 4096


@pytest.mark.slow
def test_communication_accounting(fed_run) -> None:
    c = fed_run["communication"]
    per = c["per_client_per_round"]
    assert c["model_bytes"] == sum(int(np.prod(s)) * 4 for _, s, _ in weight_spec(make_model_for_test()))
    assert c["model_bytes"] < per["train_download"] < c["model_bytes"] * 1.1
    assert c["model_bytes"] < per["train_upload"] < c["model_bytes"] * 1.1
    assert c["total_bytes"] == sum(c["totals"].values())
    assert c["total_train_bytes"] == pytest.approx(2 * 3 * (per["train_download"] + per["train_upload"]))
    assert fed_run["history"][-1]["bytes_cumulative"] == c["total_bytes"]
    assert c["convergence"]["rounds"] in (1, 2)


def make_model_for_test():
    from tbml.config import load_config

    return make_model(load_config(profile="test"), len(FEATURE_NAMES))


@pytest.mark.slow
def test_model_hashes_are_recorded(fed_run) -> None:
    audit = fed_run["model_hashes"]
    assert len(audit) == 2 * (3 + 1), "per round: one hash per client update plus the global model"
    assert {a["kind"] for a in audit} == {"client_update", "global_model"}
    assert all(len(a["sha256"]) == 64 for a in audit)
    assert len({a["sha256"] for a in audit}) == len(audit), "every update and every global model is distinct"
    assert sorted(a["client"] for a in audit if a["kind"] == "client_update" and a["round"] == 1) == [0, 1, 2]


@pytest.mark.slow
def test_global_model_is_saved_with_the_federated_scaler(cfg, fed_run, scenario, clients) -> None:
    _, parts = clients
    model, meta = load_global_model(cfg, 0, scenario, "fedavg", tag="test")
    assert meta["best_round"] == fed_run["best_round"] and meta["feature_names"] == FEATURE_NAMES
    # the scaler built from per-client sums equals the scaler of the pooled training rows
    pooled = np.concatenate([p.part("train").X for p in parts])
    mean, std = moments_to_scaler(*feature_moments(pooled))
    np.testing.assert_allclose(model.scaler.mean.numpy(), mean, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(model.scaler.std.numpy(), std, rtol=1e-4, atol=1e-5)
    # the saved model reproduces the federated test metrics when applied to the clients' rows
    test = [p.part("test") for p in parts]
    scores = np.concatenate([predict_proba(model, t.X) for t in test])
    typ = np.concatenate([t.typology for t in test])
    hist = ScoreHist.from_scores(scores, typ, int(cfg.fl.hist_bins))
    assert hist.at(meta["threshold_bin"])["confusion"] == fed_run["final"]["confusion"]


@pytest.mark.slow
def test_fedprox_runs(cfg, clients, scenario) -> None:
    r = run_federated(cfg, 0, scenario, "fedprox", rounds=1, save=False)
    assert r["settings"]["proximal_mu"] == pytest.approx(float(cfg.fl.proximal_mu))
    assert r["boundary"]["violations"] == 0 and len(r["history"]) == 1


def test_local_only_and_fragmentation_diagnostic(cfg, clients, scenario) -> None:
    local = run_local_only(cfg, 0, scenario, save=False)
    assert len(local["per_client"]) == 3
    conf = local["final"]["confusion"]
    assert sum(conf.values()) == sum(c["n_test"] for c in local["per_client"])
    assert 0.0 <= local["final"]["f1"] <= 1.0
    pooled = run_pooled_local_features(cfg, 0, scenario, save=False)
    assert pooled["final"]["n"] == local["final"]["n"], "both are measured on the same test trades"
    assert pooled["final"]["pr_auc"] > 3 * pooled["final"]["prevalence"]
