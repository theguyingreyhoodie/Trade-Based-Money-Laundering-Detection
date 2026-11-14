"""Run one federated experiment with Flower's simulation runtime (Step 04, objective 3)."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from flwr.app import Context
from flwr.serverapp import Grid, ServerApp
from flwr.simulation import run_simulation

from tbml.config import Config
from tbml.eval.metrics import N_CLASSES
from tbml.features.dictionary import FEATURE_NAMES
from tbml.fl.boundary import BoundarySpec
from tbml.fl.client import make_client_app
from tbml.fl.data import prepare_clients, run_fingerprint, scenario_dir
from tbml.fl.extensions import N_SLOT
from tbml.fl.partition import Scenario
from tbml.fl.server import AuditSink, RunState, make_strategy, run_rounds
from tbml.models.mlp import get_weights, make_model, trainable_mask, weight_spec
from tbml.utils import derive_seed, get_logger, save_json

log = get_logger(__name__)


def result_path(cfg: Config, seed: int, scenario: Scenario, strategy: str, tag: str = "") -> Path:
    name = f"{scenario.name}__{strategy}{('_' + tag) if tag else ''}.json"
    return cfg.path("results_dir", "fl", f"seed_{seed}", name)


def model_dir(cfg: Config, seed: int, scenario: Scenario, strategy: str, tag: str = "") -> Path:
    return scenario_dir(cfg, seed, scenario) / f"{strategy}{('_' + tag) if tag else ''}"


def communication_summary(state: RunState, history: list[dict[str, Any]], n_clients: int, target_f1: float | None) -> dict[str, Any]:
    """Bytes moved per client per round, in total, and until a target F1 is first reached."""
    totals: dict[str, int] = {}
    for row in state.comm:
        key = f"{row['phase']}_{row['direction']}"
        totals[key] = totals.get(key, 0) + row["bytes"]
    rounds = max(len(history), 1)
    n_train_msgs = max(sum(1 for r in state.comm if r["phase"] == "train" and r["direction"] == "up"), 1)
    n_eval_msgs = max(sum(1 for r in state.comm if r["phase"] == "evaluate" and r["direction"] == "up"), 1)
    out: dict[str, Any] = {
        "model_bytes": state.spec.model_bytes(),
        "n_parameters": int(sum(int(np.prod(shape)) for (_, shape, _), t in zip(state.spec.arrays, state.trainable) if t)),
        "per_client_per_round": {
            "train_download": totals.get("train_down", 0) / n_train_msgs,
            "train_upload": totals.get("train_up", 0) / n_train_msgs,
            "eval_download": totals.get("evaluate_down", 0) / n_eval_msgs,
            "eval_upload": totals.get("evaluate_up", 0) / n_eval_msgs,
        },
        "per_round_all_clients": {"train": (totals.get("train_down", 0) + totals.get("train_up", 0)) / rounds,
                                  "evaluate": (totals.get("evaluate_down", 0) + totals.get("evaluate_up", 0)) / rounds},
        "totals": totals,
        "total_bytes": int(sum(totals.values())),
        "total_train_bytes": int(totals.get("train_down", 0) + totals.get("train_up", 0)),
        "setup_bytes": int(totals.get("stats_down", 0) + totals.get("stats_up", 0)),
        "target_f1": target_f1,
        "rounds_to_target": None,
        "bytes_to_target": None,
        "train_bytes_to_target": None,
    }
    if target_f1 is not None:
        train_cum = 0
        for row in history:
            train_cum += row["bytes_train_down"] + row["bytes_train_up"]
            if row["test_f1"] >= target_f1:
                out.update({"rounds_to_target": row["round"], "bytes_to_target": row["bytes_cumulative"], "train_bytes_to_target": train_cum})
                break
    # convergence cost: traffic until the run first reaches 95% of the best F1 it ever reaches
    if history:
        own_target = 0.95 * max(row["test_f1"] for row in history)
        first = next(row for row in history if row["test_f1"] >= own_target)
        out["convergence"] = {"f1_level": own_target, "rounds": first["round"], "bytes": first["bytes_cumulative"]}
    return out


def run_federated(
    cfg: Config,
    seed: int,
    scenario: Scenario,
    strategy: str = "fedavg",
    rounds: int | None = None,
    audit_sink: AuditSink | None = None,
    options: dict[str, Any] | None = None,
    target_f1: float | None = None,
    tag: str = "",
    save: bool = True,
    keep_weights: bool = False,
) -> dict[str, Any]:
    """Simulate one federation and return (and optionally save) its results.

    ``keep_weights`` adds the selected global model's arrays to the returned
    dict under ``"weights"`` (they are never written to the result JSON).

    ``options`` switches on the extensions: ``{"secagg": {...}}``,
    ``{"dp": {...}}``, ``{"poison": {...}}``, ``{"norm_clip": float}``.
    ``audit_sink`` receives every model-hash audit entry as it is produced
    (used to register hashes on the evidence ledger).
    """
    options = dict(options or {})
    rounds = int(cfg.fl.rounds if rounds is None else rounds)
    partition = prepare_clients(cfg, seed, scenario)
    root = scenario_dir(cfg, seed, scenario)
    n_clients = scenario.n_clients
    n_features = len(FEATURE_NAMES)

    template = make_model(cfg, n_features, seed=derive_seed(seed, "fl-init") % (2**31))
    names = [name for name, _, _ in weight_spec(template)]
    secagg = options.get("secagg")
    spec = BoundarySpec(
        arrays=weight_spec(template),
        n_features=n_features,
        hist_len=N_CLASSES * int(cfg.fl.hist_bins),
        array_dtype_override="uint64" if secagg else None,
        extra_array_names=(N_SLOT,) if secagg else (),
    )
    clip = options.get("norm_clip", cfg.fl.norm_clip)
    state = RunState(cfg=cfg, spec=spec, names=names, trainable=trainable_mask(template), n_clients=n_clients, audit_sink=audit_sink,
                     norm_clip=clip if clip == "median" else float(clip), secagg=secagg)
    train_config: dict[str, Any] = {"lr": float(cfg.fl.lr), "local-epochs": int(cfg.fl.local_epochs)}
    if secagg:
        train_config["secagg-participants"] = list(range(n_clients))

    holder: dict[str, Any] = {}
    server_app = ServerApp()

    @server_app.main()
    def main(grid: Grid, context: Context) -> None:
        try:
            strat = make_strategy(cfg, strategy, n_clients, state)
            holder["out"] = run_rounds(grid, strat, state, get_weights(template), rounds, train_config, seed=derive_seed(seed, "fl-sampling"))
        except Exception as exc:  # surfaced after the simulation has shut down
            holder["error"] = exc
            holder["traceback"] = traceback.format_exc()

    client_options = {k: v for k, v in options.items() if k in ("secagg", "dp", "poison")}
    if client_options.get("dp"):
        client_options["dp"] = {**client_options["dp"], "rounds": rounds}
    client_app = make_client_app(cfg, root, seed, client_options)
    # Flower logs every round at INFO and a deprecation notice for run_simulation at WARNING
    # (see README, "Known limitations"); our own logger reports the rounds.
    logging.getLogger("flwr").setLevel(logging.ERROR)
    backend = {
        "client_resources": {"num_cpus": int(cfg.fl.backend.client_cpus), "num_gpus": 0.0},
        "init_args": {"num_cpus": max(int(cfg.fl.backend.init_cpus), n_clients), "include_dashboard": False, "log_to_driver": False},
    }
    run_simulation(server_app=server_app, client_app=client_app, num_supernodes=n_clients, backend_config=backend)
    if "error" in holder:
        log.error("federated run failed:\n%s", holder.get("traceback", ""))
        raise holder["error"]
    if "out" not in holder:
        raise RuntimeError("the Flower simulation ended without producing a result")

    out = holder["out"]
    best = out["best"]
    k = int(best["metrics"]["threshold_bin"])
    per_client = []
    for client in sorted(best["client_hists"]):
        h_val, h_test = best["client_hists"][client]
        m = h_test.at(k)
        per_client.append({"client": client, "n_test": h_test.n, "n_illicit_test": h_test.n_pos, "f1": m["f1"], "precision": m["precision"],
                           "recall": m["recall"], "fpr": m["fpr"]})

    # Variant: same global model, but every client picks its own threshold on its own
    # validation rows (needs no extra communication; done here from the per-client histograms).
    from tbml.fl.local import pooled_from_counts

    metric = str(cfg.models.threshold_metric)
    flagged = np.zeros(N_CLASSES, dtype=np.int64)
    support = np.zeros(N_CLASSES, dtype=np.int64)
    for h_val, h_test in best["client_hists"].values():
        k_own = h_val.best_threshold(metric)
        flagged += h_test.counts[:, k_own:].sum(axis=1)
        support += h_test.counts.sum(axis=1)
    client_thresholds = pooled_from_counts(flagged, support, best["hist_test"])

    result: dict[str, Any] = {
        "seed": seed,
        "fingerprint": run_fingerprint(cfg, seed),
        "scenario": scenario.to_dict(),
        "strategy": strategy,
        "tag": tag,
        "rounds": rounds,
        "n_clients": n_clients,
        "settings": {"lr": float(cfg.fl.lr), "lr_decay": float(cfg.fl.lr_decay), "local_epochs": int(cfg.fl.local_epochs), "fraction_train": float(cfg.fl.fraction_train),
                     "proximal_mu": float(cfg.fl.proximal_mu) if strategy == "fedprox" else 0.0, "norm_clip": state.norm_clip,
                     "secure_aggregation": bool(secagg), "dp": options.get("dp"), "poison": options.get("poison")},
        "partition": {"size_imbalance": partition["size_imbalance"], "illicit_rate_range": partition["illicit_rate_range"],
                      "clients": [{"client": c["client"], "n": c["n"], "illicit_rate": c["illicit_rate"], "n_typologies_present": c["n_typologies_present"]}
                                  for c in partition["clients"]]},
        "n_train_examples": out["n_train_examples"],
        "best_round": best["round"],
        "final": best["metrics"],
        "final_client_thresholds": client_thresholds,
        "last_round": out["last"],
        "per_client": per_client,
        "history": state.history,
        "communication": communication_summary(state, state.history, n_clients, target_f1),
        "boundary": {"replies_checked": len(spec.received), "violations": 0,
                     "kinds": {kind: sum(1 for r in spec.received if r["kind"] == kind) for kind in ("stats", "train", "evaluate")},
                     "max_reply_bytes": {kind: max((r["bytes"] for r in spec.received if r["kind"] == kind), default=0) for kind in ("stats", "train", "evaluate")},
                     "array_records_outside_training": sum(r["n_arrays"] for r in spec.received if r["kind"] != "train")},
        "model_hashes": state.audit,
        "update_norms": {"clipped_updates": state.clipped, "history": state.norm_history},
        "dp_clients": {str(c): v for c, v in sorted(state.dp_report.items())},
        "seconds": out["seconds"],
    }
    if save:
        mdir = model_dir(cfg, seed, scenario, strategy, tag)
        mdir.mkdir(parents=True, exist_ok=True)
        np.savez(mdir / "global_model.npz", **{name: w for name, w in zip(names, best["weights"])})
        save_json({"best_round": best["round"], "threshold": best["metrics"]["threshold"], "threshold_bin": k, "feature_names": FEATURE_NAMES,
                   "hist_val": best["hist_val"].to_list(), "hist_bins": int(cfg.fl.hist_bins), "scenario": scenario.to_dict(), "strategy": strategy,
                   "seed": seed}, mdir / "model_meta.json")
        save_json(result, result_path(cfg, seed, scenario, strategy, tag))
    if keep_weights:
        result["weights"] = {name: w for name, w in zip(names, best["weights"])}
    final = result["final"]
    log.info("seed %d %s %s: best round %d  F1 %.3f  P %.3f  R %.3f  PR-AUC %.3f  (%.0fs, %.1f MB total)", seed, scenario.name, strategy + (f"/{tag}" if tag else ""),
             best["round"], final["f1"], final["precision"], final["recall"], final["pr_auc"], out["seconds"], result["communication"]["total_bytes"] / 1e6)
    return result


def load_global_model(cfg: Config, seed: int, scenario: Scenario, strategy: str = "fedavg", tag: str = ""):
    """Rebuild the saved global model; returns ``(model, meta)``."""
    from tbml.models.mlp import set_weights
    from tbml.utils import load_json

    mdir = model_dir(cfg, seed, scenario, strategy, tag)
    meta = load_json(mdir / "model_meta.json")
    model = make_model(cfg, len(meta["feature_names"]))
    with np.load(mdir / "global_model.npz") as z:
        set_weights(model, [z[name] for name, _, _ in weight_spec(model)])
    return model, meta
