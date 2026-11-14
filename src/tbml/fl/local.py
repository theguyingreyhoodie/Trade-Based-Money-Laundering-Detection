"""Local-only baseline: every client trains and decides on its own.

This is the status quo without collaboration. Each client fits the MLP on its
own training rows, picks its own threshold on its own validation rows and
scores its own test rows. The pooled result adds up the clients' confusion
counts, so it is measured on exactly the same test trades as the federated and
centralised models.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tbml.config import Config
from tbml.data.typologies import TYPOLOGY_CLASSES
from tbml.eval.metrics import N_CLASSES, ScoreHist
from tbml.fl.data import load_client, prepare_clients, scenario_dir
from tbml.fl.partition import Scenario
from tbml.models.mlp import fit_mlp, predict_proba
from tbml.utils import derive_seed, get_logger, save_json

log = get_logger(__name__)


def pooled_from_counts(flagged: np.ndarray, support: np.ndarray, pooled_hist: ScoreHist) -> dict[str, Any]:
    """Metrics from per-class flagged counts and supports summed over clients."""
    tp, fp = int(flagged[1:].sum()), int(flagged[0])
    fn, tn = int(support[1:].sum()) - tp, int(support[0]) - fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "alert_rate": (tp + fp) / max(int(support.sum()), 1),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "recall_by_typology": {TYPOLOGY_CLASSES[c]: (float(flagged[c] / support[c]) if support[c] else None) for c in range(1, N_CLASSES)},
        "support_by_typology": {TYPOLOGY_CLASSES[c]: int(support[c]) for c in range(1, N_CLASSES)},
        # ranking quality when every client's alerts are merged by its own score
        "pr_auc": pooled_hist.pr_auc(),
        "roc_auc": pooled_hist.roc_auc(),
        "n": int(support.sum()),
        "n_illicit": int(support[1:].sum()),
        "prevalence": float(support[1:].sum() / max(support.sum(), 1)),
    }


def run_local_only(cfg: Config, seed: int, scenario: Scenario, save: bool = True) -> dict[str, Any]:
    """Train one model per client on its own data; return pooled and per-client metrics."""
    prepare_clients(cfg, seed, scenario)
    root = scenario_dir(cfg, seed, scenario)
    n_bins = int(cfg.fl.hist_bins)
    metric = str(cfg.models.threshold_metric)
    flagged = np.zeros(N_CLASSES, dtype=np.int64)
    support = np.zeros(N_CLASSES, dtype=np.int64)
    pooled = ScoreHist.empty(n_bins)
    clients = []
    for client in range(scenario.n_clients):
        fs = load_client(root, client)
        train, val, test = fs.part("train"), fs.part("val"), fs.part("test")
        model, info = fit_mlp(cfg, train.X, train.y, val.X, val.y, seed=derive_seed(seed, "local-only", scenario.name, client) % (2**31))
        h_val = ScoreHist.from_scores(predict_proba(model, val.X), val.typology, n_bins)
        h_test = ScoreHist.from_scores(predict_proba(model, test.X), test.typology, n_bins)
        k = h_val.best_threshold(metric)
        m = h_test.at(k)
        flagged += h_test.counts[:, k:].sum(axis=1)
        support += h_test.counts.sum(axis=1)
        pooled = pooled + h_test
        clients.append({"client": client, "n_train": len(train), "n_illicit_train": int(train.y.sum()), "n_test": h_test.n,
                        "n_illicit_test": h_test.n_pos, "f1": m["f1"], "precision": m["precision"], "recall": m["recall"], "fpr": m["fpr"],
                        "pr_auc": h_test.pr_auc(), "threshold": m["threshold"], "best_epoch": info["best_epoch"]})
    result = {
        "seed": seed,
        "scenario": scenario.to_dict(),
        "strategy": "local",
        "n_clients": scenario.n_clients,
        "final": pooled_from_counts(flagged, support, pooled),
        "per_client": clients,
        "client_f1_range": [float(min(c["f1"] for c in clients)), float(max(c["f1"] for c in clients))],
    }
    if save:
        save_json(result, cfg.path("results_dir", "fl", f"seed_{seed}", f"{scenario.name}__local.json"))
    final = result["final"]
    log.info("seed %d %s local-only: pooled F1 %.3f  P %.3f  R %.3f  (client F1 %.2f-%.2f)", seed, scenario.name, final["f1"], final["precision"],
             final["recall"], *result["client_f1_range"])
    return result


def run_pooled_local_features(cfg: Config, seed: int, scenario: Scenario, save: bool = True) -> dict[str, Any]:
    """Diagnostic: one model trained on the clients' locally computed features, pooled.

    This is not a deployable method (it pools rows). It separates the two reasons
    a federation can fall short of the centralised model:

    * *information fragmentation*: every client computes its context indicators
      (entity history, trade graph, B/L sharing) from its own trades only. This
      baseline pays that price but none other, so the gap from the centralised
      model to this one is what fragmentation costs.
    * *federated optimisation*: the gap from this baseline to FedAvg is what
      training by parameter averaging costs.
    """
    from tbml.eval.metrics import evaluate_hists

    prepare_clients(cfg, seed, scenario)
    root = scenario_dir(cfg, seed, scenario)
    parts = [load_client(root, c) for c in range(scenario.n_clients)]
    X = np.concatenate([p.X for p in parts])
    y = np.concatenate([p.y for p in parts])
    typ = np.concatenate([p.typology for p in parts])
    split = np.concatenate([p.split for p in parts])
    tr, va, te = split == "train", split == "val", split == "test"
    model, info = fit_mlp(cfg, X[tr], y[tr], X[va], y[va], seed=derive_seed(seed, "pooled-local", scenario.name) % (2**31))
    n_bins = int(cfg.fl.hist_bins)
    h_val = ScoreHist.from_scores(predict_proba(model, X[va]), typ[va], n_bins)
    h_test = ScoreHist.from_scores(predict_proba(model, X[te]), typ[te], n_bins)
    result = {"seed": seed, "scenario": scenario.to_dict(), "strategy": "pooled_local_features", "n_clients": scenario.n_clients,
              "final": evaluate_hists(h_val, h_test, str(cfg.models.threshold_metric)), "best_epoch": info["best_epoch"]}
    if save:
        save_json(result, cfg.path("results_dir", "fl", f"seed_{seed}", f"{scenario.name}__pooled_local_features.json"))
    final = result["final"]
    log.info("seed %d %s pooled model on local features: F1 %.3f  P %.3f  R %.3f  PR-AUC %.3f", seed, scenario.name, final["f1"],
             final["precision"], final["recall"], final["pr_auc"])
    return result


def run_personalised(cfg: Config, seed: int, scenario: Scenario, strategy: str = "fedavg", tag: str = "", save: bool = True) -> dict[str, Any]:
    """FedAvg followed by local fine-tuning: each client adapts the global model to its own data.

    Starting from the global model (weights and shared scaler), a client trains
    a few more epochs on its own rows with a small learning rate, keeps whichever
    version validates best on its own validation rows (possibly the unchanged
    global model), and picks its own threshold. Nothing is sent anywhere, so this
    costs no communication and no privacy beyond the federated run itself.
    """
    from tbml.fl.simulation import load_global_model
    from tbml.models.mlp import get_weights

    model, meta = load_global_model(cfg, seed, scenario, strategy, tag)
    start = get_weights(model)
    root = scenario_dir(cfg, seed, scenario)
    n_bins = int(cfg.fl.hist_bins)
    metric = str(cfg.models.threshold_metric)
    p = cfg.fl.personalisation
    flagged = np.zeros(N_CLASSES, dtype=np.int64)
    support = np.zeros(N_CLASSES, dtype=np.int64)
    pooled = ScoreHist.empty(n_bins)
    clients = []
    for client in range(scenario.n_clients):
        fs = load_client(root, client)
        train, val, test = fs.part("train"), fs.part("val"), fs.part("test")
        tuned, info = fit_mlp(cfg, train.X, train.y, val.X, val.y, seed=derive_seed(seed, "personalise", scenario.name, client) % (2**31),
                              epochs=int(p.epochs), init_weights=start, lr=float(p.lr))
        h_val = ScoreHist.from_scores(predict_proba(tuned, val.X), val.typology, n_bins)
        h_test = ScoreHist.from_scores(predict_proba(tuned, test.X), test.typology, n_bins)
        k = h_val.best_threshold(metric)
        m = h_test.at(k)
        flagged += h_test.counts[:, k:].sum(axis=1)
        support += h_test.counts.sum(axis=1)
        pooled = pooled + h_test
        clients.append({"client": client, "f1": m["f1"], "precision": m["precision"], "recall": m["recall"], "kept_epoch": info["best_epoch"],
                        "n_test": h_test.n, "n_illicit_test": h_test.n_pos})
    result = {"seed": seed, "scenario": scenario.to_dict(), "strategy": f"{strategy}+finetune", "n_clients": scenario.n_clients,
              "global_round": meta["best_round"], "final": pooled_from_counts(flagged, support, pooled), "per_client": clients,
              "clients_that_changed_the_model": sum(c["kept_epoch"] > 0 for c in clients)}
    if save:
        save_json(result, cfg.path("results_dir", "fl", f"seed_{seed}", f"{scenario.name}__{strategy}_finetuned.json"))
    log.info("seed %d %s %s + local fine-tuning: pooled F1 %.3f  P %.3f  R %.3f", seed, scenario.name, strategy, result["final"]["f1"],
             result["final"]["precision"], result["final"]["recall"])
    return result
