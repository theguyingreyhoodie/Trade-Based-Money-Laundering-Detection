"""Stretch experiments: secure aggregation, DP-SGD, poisoning defence, fine-tuning, vertical FL."""

from __future__ import annotations

from typing import Any

import numpy as np

from tbml.config import Config
from tbml.data.generator import load_dataset
from tbml.features.pipeline import load_central_features
from tbml.fl.data import cached_result, run_fingerprint
from tbml.fl.experiments import experiment_plan, primary_scenario, target_f1
from tbml.fl.local import run_personalised
from tbml.fl.partition import Scenario
from tbml.fl.simulation import model_dir, result_path, run_federated
from tbml.fl.vertical import run_vertical_demo
from tbml.utils import get_logger

log = get_logger(__name__)

SECAGG_SECRET = "pairwise-keys-agreed-among-clients"  # stands in for Diffie-Hellman key agreement; never read by server code


def _reference(cfg: Config, seed: int, scenario: Scenario, strategy: str = "fedavg") -> dict[str, Any]:
    """The plain run of the sweep for the same seed and scenario (run it if it is missing)."""
    saved = cached_result(result_path(cfg, seed, scenario, strategy), run_fingerprint(cfg, seed))
    return saved or run_federated(cfg, seed, scenario, strategy, target_f1=target_f1(cfg, seed))


def _brief(result: dict[str, Any]) -> dict[str, Any]:
    final = result["final"]
    return {"f1": final["f1"], "precision": final["precision"], "recall": final["recall"], "pr_auc": final["pr_auc"], "best_round": result["best_round"],
            "last_round_f1": result["last_round"]["f1"],
            "total_bytes": result["communication"]["total_bytes"], "train_upload_per_client_round": result["communication"]["per_client_per_round"]["train_upload"]}


def secure_aggregation(cfg: Config, seed: int) -> dict[str, Any]:
    """FedAvg with pairwise additive masking against plain FedAvg, same seed and scenario.

    Two comparisons: after exactly one round, where both start from the same
    model and must agree up to rounding; and over the full run, where tiny
    rounding differences are amplified by training and the curves drift apart
    slightly.
    """
    scenario = primary_scenario(cfg)
    options = {"secagg": {"group_secret": SECAGG_SECRET}}
    one_plain = run_federated(cfg, seed, scenario, "fedavg", rounds=1, save=False, keep_weights=True)
    one_masked = run_federated(cfg, seed, scenario, "fedavg", rounds=1, options=options, save=False, keep_weights=True)
    one_round_diff = max(float(np.abs(one_plain["weights"][k].astype(np.float64) - one_masked["weights"][k].astype(np.float64)).max()) for k in one_plain["weights"])

    plain = _reference(cfg, seed, scenario)
    masked = run_federated(cfg, seed, scenario, "fedavg", options=options, tag="secagg")
    curve_gap = max(abs(x["test_f1"] - y["test_f1"]) for x, y in zip(plain["history"], masked["history"]))
    return {
        "seed": seed, "scenario": scenario.to_dict(),
        "one_round": {"max_abs_weight_difference": one_round_diff, "same_confusion_matrix": one_plain["final"]["confusion"] == one_masked["final"]["confusion"],
                      "f1_plain": one_plain["final"]["f1"], "f1_secure": one_masked["final"]["f1"]},
        "plain": _brief(plain), "secure": _brief(masked),
        "max_f1_difference_over_rounds": curve_gap,
        "upload_overhead_factor": masked["communication"]["per_client_per_round"]["train_upload"] / plain["communication"]["per_client_per_round"]["train_upload"],
        "boundary": masked["boundary"],
        "note": "updates are fixed-point encoded in 64-bit integers and masked pairwise; the server only ever sees masked arrays and their sum",
    }


def differential_privacy(cfg: Config, seeds: list[int]) -> dict[str, Any]:
    """Privacy / utility curve: FedAvg with DP-SGD clients at several epsilon values."""
    scenario = primary_scenario(cfg)
    d = cfg.fl.dp
    rows = []
    for eps in [float(e) for e in d.epsilons]:
        runs = []
        for seed in seeds:
            options = {"dp": {"epsilon": eps, "delta": float(d.delta), "max_grad_norm": float(d.max_grad_norm)}}
            r = run_federated(cfg, seed, scenario, "fedavg", options=options, tag=f"dp{eps:g}", save=False)
            spent = [c["epsilon"] for c in r["dp_clients"].values()]
            runs.append({"seed": seed, **_brief(r), "epsilon_spent_max": max(spent), "noise_multipliers": [c["noise_multiplier"] for c in r["dp_clients"].values()]})
        rows.append({"epsilon": eps, "delta": float(d.delta), "max_grad_norm": float(d.max_grad_norm), "runs": runs})
    reference = [{"seed": seed, **_brief(_reference(cfg, seed, scenario))} for seed in seeds]
    return {"scenario": scenario.to_dict(), "seeds": seeds, "private": rows, "non_private": reference,
            "note": "example-level (epsilon, delta)-DP for each client's own training rows, accounted with Renyi DP over all local steps of the run"}


def poisoning(cfg: Config, seeds: list[int]) -> dict[str, Any]:
    """One malicious client out of three, with and without norm clipping at the server."""
    scenario = primary_scenario(cfg)
    attack = cfg.fl.poison.to_dict()
    settings = {
        "attack_no_defence": {"poison": attack},
        "attack_median_clipping": {"poison": attack, "norm_clip": "median"},
        "no_attack_median_clipping": {"norm_clip": "median"},
    }
    out: dict[str, Any] = {"scenario": scenario.to_dict(), "seeds": seeds, "attack": attack,
                           "runs": {"no_attack": [{"seed": s, **_brief(_reference(cfg, s, scenario))} for s in seeds]}}
    for name, options in settings.items():
        runs = []
        for seed in seeds:
            r = run_federated(cfg, seed, scenario, "fedavg", options=options, tag=name, save=False)
            norms = r["update_norms"]["history"]
            runs.append({"seed": seed, **_brief(r), "clipped_updates": r["update_norms"]["clipped_updates"],
                         "round1_norms": norms[0]["norms"] if norms else {}})
        out["runs"][name] = runs
    return out


def personalisation(cfg: Config, seeds: list[int]) -> dict[str, Any]:
    """Local fine-tuning of the FedAvg model on every scenario of the sweep."""
    rows = []
    for scenario, _ in experiment_plan(cfg):
        for seed in seeds:
            _reference(cfg, seed, scenario)
            r = run_personalised(cfg, seed, scenario)
            rows.append({"seed": seed, "scenario": scenario.name, "f1": r["final"]["f1"], "precision": r["final"]["precision"], "recall": r["final"]["recall"],
                         "clients_that_changed_the_model": r["clients_that_changed_the_model"], "n_clients": scenario.n_clients})
    return {"runs": rows}


def vertical(cfg: Config, seeds: list[int]) -> dict[str, Any]:
    """Split-NN demo for the Bank / Trade registry / Customs picture."""
    return {"runs": [run_vertical_demo(cfg, seed, load_dataset(cfg.seed_dir("data_dir", seed)), load_central_features(cfg, seed)) for seed in seeds]}
