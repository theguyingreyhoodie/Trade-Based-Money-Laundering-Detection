"""MLP building blocks and the centralised baselines."""

from __future__ import annotations

import numpy as np
import torch

from tbml.models.baselines import load_baseline_scores, run_and_save, run_baselines
from tbml.models.mlp import (
    TBMLNet,
    feature_moments,
    fit_mlp,
    get_weights,
    make_model,
    moments_to_scaler,
    predict_proba,
    set_scaler,
    set_weights,
    train_epochs,
    trainable_mask,
    weight_spec,
)


def _toy(n: int = 1500, d: int = 6, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32) * np.array([1, 10, 100, 1, 1, 0.1], dtype=np.float32) + 5
    y = ((X[:, 0] - 5) + 0.05 * (X[:, 1] - 5) > 1.2).astype(np.int8)
    return X, y


def test_weights_round_trip(cfg) -> None:
    model = make_model(cfg, 6, seed=1)
    weights = get_weights(model)
    spec = weight_spec(model)
    assert [tuple(w.shape) for w in weights] == [shape for _, shape, _ in spec]
    assert [name for name, _, _ in spec][:2] == ["scaler.mean", "scaler.std"]
    assert trainable_mask(model)[:2] == [False, False] and all(trainable_mask(model)[2:])

    other = make_model(cfg, 6, seed=2)
    assert any(not np.array_equal(a, b) for a, b in zip(get_weights(other), weights))
    set_weights(other, weights)
    for a, b in zip(get_weights(other), weights):
        np.testing.assert_array_equal(a, b)
    # same seed -> same initialisation
    for a, b in zip(get_weights(make_model(cfg, 6, seed=1)), weights):
        np.testing.assert_array_equal(a, b)


def test_scaler_from_moments_matches_numpy_and_aggregates() -> None:
    X, _ = _toy()
    mean, std = moments_to_scaler(*feature_moments(X))
    np.testing.assert_allclose(mean, X.mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(std, X.std(axis=0), rtol=1e-4)
    # moments of the parts add up to the moments of the whole: the basis of federated standardisation
    parts = np.array_split(X, 4)
    n = sum(feature_moments(p)[0] for p in parts)
    s = sum(feature_moments(p)[1] for p in parts)
    q = sum(feature_moments(p)[2] for p in parts)
    mean2, std2 = moments_to_scaler(n, s, q)
    np.testing.assert_allclose(mean2, mean, rtol=1e-6)
    np.testing.assert_allclose(std2, std, rtol=1e-6)
    # a constant feature must not produce a zero divisor
    const = np.ones((10, 2), dtype=np.float32)
    assert (moments_to_scaler(*feature_moments(const))[1] == 1.0).all()


def test_standardisation_is_part_of_the_model(cfg) -> None:
    X, _ = _toy()
    model = make_model(cfg, 6, seed=0)
    with torch.no_grad():
        set_scaler(model, *moments_to_scaler(*feature_moments(X)))
    z = model.scaler(torch.as_tensor(X)).numpy()
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1.0) < 0.05
    assert isinstance(model, TBMLNet) and model(torch.as_tensor(X)).shape == (len(X),)


def test_training_learns_and_is_deterministic(cfg) -> None:
    X, y = _toy()

    def run(seed: int) -> tuple[list[np.ndarray], float]:
        model = make_model(cfg, 6, seed=seed)
        with torch.no_grad():
            set_scaler(model, *moments_to_scaler(*feature_moments(X)))
        first = train_epochs(model, X, y, cfg, 1, seed=seed)["loss"]
        last = train_epochs(model, X, y, cfg, 5, seed=seed + 1)["loss"]
        assert last < first
        return get_weights(model), last

    w1, _ = run(3)
    w2, _ = run(3)
    for a, b in zip(w1, w2):
        np.testing.assert_array_equal(a, b)
    p = predict_proba(make_model(cfg, 6, seed=0), X)
    assert p.shape == (len(X),) and ((p >= 0) & (p <= 1)).all()


def test_fedprox_term_limits_drift(cfg) -> None:
    X, y = _toy()

    def drift(mu: float) -> float:
        model = make_model(cfg, 6, seed=5)
        with torch.no_grad():
            set_scaler(model, *moments_to_scaler(*feature_moments(X)))
        before = [p.detach().clone() for p in model.parameters()]
        train_epochs(model, X, y, cfg, 3, seed=1, proximal_mu=mu)
        return float(sum(((p.detach() - b) ** 2).sum() for p, b in zip(model.parameters(), before)))

    assert drift(1.0) < drift(0.0)


def test_fit_mlp_early_stopping(cfg) -> None:
    X, y = _toy(3000)
    model, info = fit_mlp(cfg, X[:2000], y[:2000], X[2000:], y[2000:], seed=0, epochs=15)
    assert 1 <= info["best_epoch"] <= info["epochs_run"] <= 15
    from sklearn.metrics import average_precision_score

    assert average_precision_score(y[2000:], predict_proba(model, X[2000:])) > 0.9


def test_baselines_on_generated_data(cfg, features, tmp_path) -> None:
    results = run_baselines(cfg, features, seed=0, out_dir=tmp_path / "baselines")
    assert set(results) == {"rules", "logreg", "random_forest", "lightgbm", "mlp"}
    prevalence = results["rules"]["prevalence"]
    for name, r in results.items():
        for key in ["precision", "recall", "f1", "pr_auc", "roc_auc", "fpr"]:
            assert 0.0 <= r[key] <= 1.0, (name, key)
        assert set(r["confusion"]) == {"tp", "fp", "fn", "tn"}
        assert sum(r["confusion"].values()) == r["n"]
        assert r["pr_auc"] > 3 * prevalence, name
    assert results["lightgbm"]["pr_auc"] > results["rules"]["pr_auc"], "a trained model must beat hand-written rules"
    assert results["lightgbm"]["top_features_by_gain"]
    for name in ["logreg.joblib", "random_forest.joblib", "lightgbm.joblib", "mlp.pt", "scores.npz"]:
        assert (tmp_path / "baselines" / name).is_file(), name


def test_run_and_save_writes_results(cfg, features) -> None:
    payload = run_and_save(cfg, features, seed=0)
    assert cfg.path("results_dir", "baselines", "seed_0.json").is_file()
    assert payload["n_features"] == features.X.shape[1]
    saved = load_baseline_scores(cfg, 0)
    assert saved["rule_hits"].shape[0] == len(features)
    assert {"score_rules", "score_lightgbm", "score_mlp", "split", "y", "typology", "trade_id"} <= set(saved)
