"""Stretch goals: secure aggregation, norm clipping, DP-SGD, fine-tuning, vertical FL."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest
import torch
from flwr.app import Array, ArrayRecord, ConfigRecord, MetricRecord, RecordDict

from tbml.eval.metrics import N_CLASSES
from tbml.features.dictionary import FEATURE_BY_NAME, FEATURE_NAMES
from tbml.fl.boundary import TRAIN, BoundarySpec, BoundaryViolation, check_reply
from tbml.fl.data import load_client, prepare_clients, scenario_dir
from tbml.fl.dp import dp_train_round, noise_for_epsilon, spent_epsilon, total_steps
from tbml.fl.extensions import N_SLOT, mask_update, mask_vector, poison_update, unmask_sum
from tbml.fl.local import run_local_only, run_personalised
from tbml.fl.partition import Scenario
from tbml.fl.server import AuditedFedAvg, RunState, to_record
from tbml.fl.simulation import run_federated
from tbml.fl.vertical import PARTIES, SplitNN, party_blocks, run_vertical_demo
from tbml.ledger.hashing import hash_arrays
from tbml.models.mlp import fit_mlp, get_weights, make_model, predict_proba, trainable_mask, weight_spec

SECRET = "test-secret"


def _updates(cfg, n_clients: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    model = make_model(cfg, 12, seed=0)
    names = [n for n, _, _ in weight_spec(model)]
    base = get_weights(model)
    weights = [[(w + rng.normal(0, 0.05, w.shape)).astype(np.float32) for w in base] for _ in range(n_clients)]
    sizes = [int(v) for v in rng.integers(200, 5000, n_clients)]
    return names, weights, sizes


# --------------------------------------------------------------------------- secure aggregation


def test_pairwise_masks_cancel_and_hide_the_update() -> None:
    rng = np.random.default_rng(0)
    vectors = [rng.normal(0, 1, 5000) for _ in range(4)]
    participants = [0, 1, 2, 3]
    masked = [mask_vector(v, c, participants, server_round=3, secret=SECRET) for c, v in enumerate(vectors)]
    with np.errstate(over="ignore"):
        total = sum(masked[1:], masked[0])
    recovered = total.view(np.int64).astype(np.float64) / 2**24
    np.testing.assert_allclose(recovered, np.sum(vectors, axis=0), atol=1e-6)

    # one masked update on its own carries no usable signal: it looks like uniform 64-bit noise
    as_float = masked[0].view(np.int64).astype(np.float64)
    assert abs(np.corrcoef(as_float, vectors[0])[0, 1]) < 0.05
    assert masked[0].dtype == np.uint64 and (masked[0] >> np.uint64(63)).mean() == pytest.approx(0.5, abs=0.03)
    # masks change every round, so two rounds cannot be differenced to remove them
    again = mask_vector(vectors[0], 0, participants, server_round=4, secret=SECRET)
    assert (again != masked[0]).mean() > 0.99
    # a different key gives different masks
    assert (mask_vector(vectors[0], 0, participants, 3, "other") != masked[0]).mean() > 0.99


def test_secure_sum_equals_weighted_average(cfg) -> None:
    names, weights, sizes = _updates(cfg)
    config = {"secagg-participants": list(range(len(weights)))}
    records = [mask_update(names, w, {"num-examples": n}, c, 5, config, {"group_secret": SECRET}) for c, (w, n) in enumerate(zip(weights, sizes))]
    for record in records:
        assert list(record.keys()) == [*names, N_SLOT]
        assert all(str(a.dtype) == "uint64" for a in record.values()), "the server receives integers, not weights"
    result = unmask_sum(records, names, {"group_secret": SECRET})
    expected = [sum(n * w[i].astype(np.float64) for w, n in zip(weights, sizes)) / sum(sizes) for i in range(len(names))]
    for got, want in zip(result, expected):
        assert got.dtype == np.float32
        np.testing.assert_allclose(got, want, atol=2e-6)

    # without one participant the masks do not cancel: the result is noise, not a model (no dropout recovery)
    try:
        broken = unmask_sum(records[:-1], names, {"group_secret": SECRET})
        assert np.abs(broken[2] - expected[2]).max() > 1.0
    except RuntimeError:
        pass


def test_boundary_in_secure_mode(cfg) -> None:
    names, weights, sizes = _updates(cfg, n_clients=2)
    model = make_model(cfg, 12, seed=0)
    spec = BoundarySpec(arrays=weight_spec(model), n_features=12, hist_len=N_CLASSES * 800, array_dtype_override="uint64", extra_array_names=(N_SLOT,))
    record = mask_update(names, weights[0], {"num-examples": sizes[0]}, 0, 1, {"secagg-participants": [0, 1]}, {"group_secret": SECRET})
    digest = hash_arrays((n, a.numpy()) for n, a in record.items())
    reply = RecordDict({"arrays": record, "metrics": MetricRecord({"num-examples": 1, "partition-id": 0, "train-loss": 0.1, "steps": 3}),
                        "meta": ConfigRecord({"update-sha256": digest})})
    receipt = check_reply(reply, spec, TRAIN)
    assert receipt["bytes"] <= spec.byte_budget(TRAIN) and receipt["n_arrays"] == len(names) + 1
    plain = RecordDict({"arrays": ArrayRecord(array_dict=OrderedDict((n, Array(w)) for n, w in zip(names, weights[0]))),
                        "metrics": MetricRecord({"num-examples": 1, "partition-id": 0}), "meta": ConfigRecord({"update-sha256": "0" * 64})})
    with pytest.raises(BoundaryViolation):
        check_reply(plain, spec, TRAIN)  # an unmasked update is refused when secure aggregation is on


# --------------------------------------------------------------------------- poisoning and clipping


def test_poison_update_scales_and_flips(cfg) -> None:
    names, weights, _ = _updates(cfg, n_clients=1)
    base = get_weights(make_model(cfg, 12, seed=0))
    local = weights[0]
    boosted = poison_update(base, local, {"mode": "boost", "scale": 5.0})
    flipped = poison_update(base, local, {"mode": "signflip", "scale": 5.0})
    np.testing.assert_allclose(boosted[3] - base[3], 5.0 * (local[3] - base[3]), atol=1e-5)
    np.testing.assert_allclose(flipped[3] - base[3], -5.0 * (local[3] - base[3]), atol=1e-5)
    np.testing.assert_array_equal(boosted[0], local[0])  # the shared scaler is left alone
    np.testing.assert_array_equal(flipped[1], local[1])


def _strategy(cfg, names, base, norm_clip):
    model = make_model(cfg, 12, seed=0)
    spec = BoundarySpec(arrays=weight_spec(model), n_features=12, hist_len=N_CLASSES * 800)
    state = RunState(cfg=cfg, spec=spec, names=names, trainable=trainable_mask(model), n_clients=3, norm_clip=norm_clip)
    state.global_weights = base
    return AuditedFedAvg(min_train_nodes=3, min_evaluate_nodes=3, min_available_nodes=3).attach(state), state


def test_median_clipping_bounds_a_boosted_update(cfg) -> None:
    names, weights, sizes = _updates(cfg, n_clients=3)
    base = get_weights(make_model(cfg, 12, seed=0))
    weights[0] = poison_update(base, weights[0], {"mode": "signflip", "scale": 20.0})
    contents = [(c, RecordDict({"arrays": to_record(names, w), "metrics": MetricRecord({"num-examples": n, "partition-id": c, "train-loss": 0.1, "steps": 1})}))
                for c, (w, n) in enumerate(zip(weights, sizes))]

    def norm(record) -> float:
        mask = trainable_mask(make_model(cfg, 12))
        return float(np.sqrt(sum(((record[n].numpy().astype(np.float64) - g) ** 2).sum() for n, g, t in zip(names, base, mask) if t)))

    strategy, state = _strategy(cfg, names, base, "median")
    clipped = dict(strategy._clip_updates(1, contents))
    honest = sorted(state.update_norms[c] for c in (1, 2))
    bound = state.norm_history[0]["bound"]
    assert state.update_norms[0] > 10 * honest[1], "the attacker's update is far larger than an honest one"
    assert honest[0] <= bound <= honest[1] + 1e-9 or bound == pytest.approx(honest[1]), "the bound is the median norm of the round"
    assert norm(clipped[0]["arrays"]) == pytest.approx(bound, rel=1e-4), "the boosted update is scaled down to the bound"
    assert state.clipped >= 1
    smallest = min((1, 2), key=lambda c: state.update_norms[c])
    np.testing.assert_array_equal(clipped[smallest]["arrays"][names[3]].numpy(), contents[smallest][1]["arrays"][names[3]].numpy())

    off, off_state = _strategy(cfg, names, base, 0.0)
    assert off._clip_updates(1, contents) == contents and off_state.clipped == 0 and off_state.norm_history[0]["bound"] is None
    fixed, fixed_state = _strategy(cfg, names, base, 0.01)
    assert all(norm(c["arrays"]) <= 0.01 * (1 + 1e-4) for _, c in fixed._clip_updates(1, contents)) and fixed_state.clipped == 3


# --------------------------------------------------------------------------- DP-SGD


def test_dp_accounting_matches_opacus() -> None:
    """The accountant must use the sampling rate and step count Opacus actually trains with."""
    import warnings

    from opacus import PrivacyEngine

    from tbml.fl.dp import sample_rate, steps_per_epoch

    for n, batch in [(4894, 256), (5000, 256), (1000, 256), (256, 256), (100, 256), (12345, 1024)]:
        dataset = torch.utils.data.TensorDataset(torch.zeros(n, 3), torch.zeros(n))
        loader = torch.utils.data.DataLoader(dataset, batch_size=min(batch, n))
        model = torch.nn.Linear(3, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, private_loader = PrivacyEngine(accountant="rdp").make_private(
                module=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1), data_loader=loader, noise_multiplier=1.0, max_grad_norm=1.0)
        assert private_loader.sample_rate == pytest.approx(sample_rate(n, batch), rel=1e-12), (n, batch)
        assert len(private_loader) == steps_per_epoch(n, batch), (n, batch)
    assert sample_rate(4894, 256) == pytest.approx(1 / 20) and steps_per_epoch(4894, 256) == 20


def test_dp_noise_calibration() -> None:
    args = dict(delta=1e-5, n=5000, batch=256, rounds=30, local_epochs=1)
    assert total_steps(5000, 256, 30, 1) == 600
    sigmas = {eps: noise_for_epsilon(eps, **args) for eps in (1.0, 3.0, 8.0)}
    assert sigmas[1.0] > sigmas[3.0] > sigmas[8.0] > 0.3, "a stricter privacy budget needs more noise"
    for eps, sigma in sigmas.items():
        assert spent_epsilon(sigma, **args) == pytest.approx(eps, rel=0.02)
    # a smaller client samples each row more often and needs more noise for the same budget
    assert noise_for_epsilon(3.0, 1e-5, 1000, 256, 30, 1) > sigmas[3.0]
    assert spent_epsilon(sigmas[3.0], 1e-5, 5000, 256, 60, 1) > 3.0, "training longer with the same noise spends more"


@pytest.fixture(scope="module")
def scenario() -> Scenario:
    return Scenario(3, "dirichlet", 0.5)


@pytest.fixture(scope="module")
def clients(cfg, dataset, scenario):
    from tbml.data.generator import write_dataset

    write_dataset(dataset, cfg.seed_dir("data_dir", 0))
    prepare_clients(cfg, 0, scenario)
    return [load_client(scenario_dir(cfg, 0, scenario), c) for c in range(3)]


def test_dp_local_training(cfg, clients) -> None:
    fs = clients[0]
    start = get_weights(make_model(cfg, fs.X.shape[1], seed=0))
    options = {"epsilon": 3.0, "delta": 1e-5, "max_grad_norm": 1.0, "rounds": 2}
    new, metrics = dp_train_round(cfg, fs, start, 0, 1, 0, lr=0.05, local_epochs=1, options=options)
    assert metrics["num-examples"] == len(fs.part("train")) and metrics["steps"] > 0
    assert metrics["dp-epsilon"] == pytest.approx(3.0, rel=0.03) and metrics["dp-noise"] > 0.3
    assert [w.shape for w in new] == [w.shape for w in start]
    assert any(not np.array_equal(a, b) for a, b in zip(new[2:], start[2:]))
    np.testing.assert_array_equal(new[0], start[0])
    assert all(np.isfinite(w).all() for w in new)


# --------------------------------------------------------------------------- simulated runs with the extensions


@pytest.mark.slow
def test_secure_aggregation_run_matches_plain_fedavg(cfg, clients, scenario) -> None:
    plain = run_federated(cfg, 0, scenario, "fedavg", rounds=1, save=False, keep_weights=True)
    secure = run_federated(cfg, 0, scenario, "fedavg", rounds=1, save=False, keep_weights=True, options={"secagg": {"group_secret": SECRET}})
    for name in plain["weights"]:
        np.testing.assert_allclose(secure["weights"][name], plain["weights"][name], atol=1e-5)
    assert secure["final"]["confusion"] == plain["final"]["confusion"]
    assert secure["boundary"]["violations"] == 0 and secure["settings"]["secure_aggregation"]
    ratio = secure["communication"]["per_client_per_round"]["train_upload"] / plain["communication"]["per_client_per_round"]["train_upload"]
    assert 1.8 < ratio < 2.2, "64-bit masked integers double the upload"


@pytest.mark.slow
def test_dp_and_poisoned_runs(cfg, clients, scenario) -> None:
    dp = run_federated(cfg, 0, scenario, "fedavg", rounds=1, save=False, options={"dp": {"epsilon": 3.0, "delta": 1e-5, "max_grad_norm": 1.0}})
    assert set(dp["dp_clients"]) == {"0", "1", "2"}
    assert all(c["epsilon"] == pytest.approx(3.0, rel=0.05) and c["noise_multiplier"] > 0 for c in dp["dp_clients"].values())
    assert dp["boundary"]["violations"] == 0

    attack = {"clients": [0], "mode": "signflip", "scale": 5.0}
    clean = run_federated(cfg, 0, scenario, "fedavg", rounds=1, save=False)
    defended = run_federated(cfg, 0, scenario, "fedavg", rounds=1, save=False, options={"poison": attack, "norm_clip": "median"})
    honest = clean["update_norms"]["history"][0]["norms"]
    record = defended["update_norms"]["history"][0]
    norms = record["norms"]
    assert norms["0"] == pytest.approx(5.0 * honest["0"], rel=0.01), "the attacker sends its update boosted five-fold"
    assert norms["1"] == pytest.approx(honest["1"], rel=1e-4) and norms["2"] == pytest.approx(honest["2"], rel=1e-4)
    assert record["bound"] == pytest.approx(float(np.median(list(norms.values()))), rel=1e-6)
    # exactly the updates above the median are scaled down: here one of three
    assert defended["update_norms"]["clipped_updates"] == 1
    assert clean["update_norms"]["clipped_updates"] == 0 and clean["update_norms"]["history"][0]["bound"] is None
    assert defended["settings"]["norm_clip"] == "median" and defended["settings"]["poison"] == attack


# --------------------------------------------------------------------------- fine-tuning


def test_warm_start_keeps_the_start_when_nothing_improves(cfg) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 6)).astype(np.float32)
    y = (X[:, 0] > 1.0).astype(np.int8)
    good, _ = fit_mlp(cfg, X[:600], y[:600], X[600:], y[600:], seed=0, epochs=20)
    start = get_weights(good)
    noise = rng.integers(0, 2, 600).astype(np.int8)  # training on noise cannot beat the good start on validation
    tuned, info = fit_mlp(cfg, X[:600], noise, X[600:], y[600:], seed=1, epochs=3, init_weights=start, lr=0.01)
    assert info["best_epoch"] == 0
    for a, b in zip(get_weights(tuned), start):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(predict_proba(tuned, X[600:]), predict_proba(good, X[600:]))


@pytest.mark.slow
def test_personalised_fine_tuning(cfg, clients, scenario) -> None:
    run_federated(cfg, 0, scenario, "fedavg", rounds=2)
    result = run_personalised(cfg, 0, scenario, save=False)
    local = run_local_only(cfg, 0, scenario, save=False)
    assert result["final"]["n"] == local["final"]["n"], "measured on the same test trades as local-only"
    assert len(result["per_client"]) == 3 and 0 <= result["clients_that_changed_the_model"] <= 3
    assert all(0 <= c["kept_epoch"] <= int(cfg.fl.personalisation.epochs) for c in result["per_client"])
    assert result["strategy"] == "fedavg+finetune" and 0.0 <= result["final"]["f1"] <= 1.0


# --------------------------------------------------------------------------- vertical FL


def test_party_blocks_hold_only_own_columns(cfg, dataset, features) -> None:
    blocks = party_blocks(dataset, features, cfg)
    assert list(blocks) == PARTIES
    seen: set[str] = set()
    for party, (X, names) in blocks.items():
        assert X.shape == (len(features), len(names)) and X.dtype == np.float32 and np.isfinite(X).all()
        assert not (seen & set(names)), "no column belongs to two parties"
        seen |= set(names)
        for name in names:
            if name in FEATURE_BY_NAME:
                assert FEATURE_BY_NAME[name].owner == party, f"{name} is not computable by the {party} party alone"
    cross = {f for f in FEATURE_NAMES if FEATURE_BY_NAME[f].owner == "cross"}
    assert cross and not (cross & seen), "indicators that need two parties' fields are not available to anyone"
    assert {"hs_mismatch", "inv_pay_gap", "customs_value_log_ratio"} <= cross


def test_split_network_trains_and_accounts_for_messages(cfg) -> None:
    rng = np.random.default_rng(0)
    n = 1024
    blocks = {"bank": rng.normal(size=(n, 4)).astype(np.float32), "trade": rng.normal(size=(n, 6)).astype(np.float32), "customs": rng.normal(size=(n, 3)).astype(np.float32)}
    # the label needs columns of two different parties: no party can learn it alone
    y = ((blocks["bank"][:, 0] + blocks["customs"][:, 0]) > 1.0).astype(np.float32)
    model = SplitNN(cfg, {p: b.shape[1] for p, b in blocks.items()}, seed=0)
    with torch.no_grad():
        model.fit_scalers(blocks)
    tensors = {p: torch.as_tensor(b) for p, b in blocks.items()}
    generator = torch.Generator().manual_seed(0)
    losses = [model.train_epoch(tensors, torch.as_tensor(y), generator) for _ in range(25)]
    assert losses[-1] < 0.6 * losses[0]
    from sklearn.metrics import average_precision_score

    assert average_precision_score(y, model.predict(blocks)) > 0.9
    # per epoch every party sends one embedding per trade and receives one gradient of the same size
    per_epoch = n * model.embedding * 4 * len(blocks)
    assert model.bytes_up == model.bytes_down == 25 * per_epoch
    state = model.state()
    before = model.predict(blocks)
    model.train_epoch(tensors, torch.as_tensor(y), generator)
    model.load(state)
    np.testing.assert_allclose(model.predict(blocks), before, atol=1e-6)


def test_vertical_demo(cfg, dataset, features) -> None:
    result = run_vertical_demo(cfg, 0, dataset, features)
    methods = result["methods"]
    assert set(methods) == {"split_nn", "bank_alone", "trade_alone", "customs_alone", "pooled_columns"}
    for name, m in methods.items():
        assert 0.0 <= m["f1"] <= 1.0 and m["pr_auc"] > m["prevalence"], name
    assert methods["split_nn"]["pr_auc"] > max(methods["bank_alone"]["pr_auc"], methods["customs_alone"]["pr_auc"]), "combining parties must beat the weaker parties alone"
    c = result["communication"]
    assert c["bytes_total"] == c["bytes_up"] + c["bytes_down"] == c["bytes_per_epoch"] * result["epochs"]
    assert c["bytes_per_trade_per_epoch"] == result["embedding"] * 4 * 2 * 3
    assert sum(p["n_features"] for p in result["parties"].values()) > 60
