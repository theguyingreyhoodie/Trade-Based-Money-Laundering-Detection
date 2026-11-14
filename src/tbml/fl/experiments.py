"""The federated experiment plan: which scenarios and methods are compared."""

from __future__ import annotations

from typing import Any

from tbml.config import Config
from tbml.fl.data import cached_result, run_fingerprint
from tbml.fl.local import run_local_only, run_pooled_local_features
from tbml.fl.partition import Scenario
from tbml.fl.simulation import result_path, run_federated
from tbml.utils import get_logger, load_json

log = get_logger(__name__)


def primary_scenario(cfg: Config) -> Scenario:
    p = cfg.fl.primary
    return Scenario(int(p.n_clients), str(p.partition), None if p.partition == "iid" else float(p.alpha))


def experiment_plan(cfg: Config) -> list[tuple[Scenario, list[str]]]:
    """Scenarios and the federated strategies to run on each.

    * base client count: IID, Dirichlet for every alpha, and quantity skew;
      FedProx is added wherever the split is non-IID;
    * other client counts: IID and the middle alpha (FedAvg), for the
      F1-versus-number-of-clients curve.
    Local-only and the pooled-local-features diagnostic run on every scenario.
    """
    e = cfg.fl.experiments
    counts = [int(n) for n in e.n_clients]
    alphas = [float(a) for a in e.alphas]
    base = counts[0]
    middle = alphas[len(alphas) // 2]
    plan: list[tuple[Scenario, list[str]]] = [(Scenario(base, "iid"), ["fedavg"])]
    plan += [(Scenario(base, "dirichlet", a), ["fedavg", "fedprox"]) for a in alphas]
    plan.append((Scenario(base, "quantity", float(e.quantity_skew_alpha)), ["fedavg", "fedprox"]))
    for n in counts[1:]:
        plan.append((Scenario(n, "iid"), ["fedavg"]))
        plan.append((Scenario(n, "dirichlet", middle), ["fedavg"]))
    return plan


def centralised_mlp_f1(cfg: Config, seed: int) -> float | None:
    """Test F1 of the centralised MLP for this seed (the reference for the communication target)."""
    path = cfg.path("results_dir", "baselines", f"seed_{seed}.json")
    if not path.is_file():
        return None
    return float(load_json(path)["methods"]["mlp"]["f1"])


def target_f1(cfg: Config, seed: int) -> float | None:
    reference = centralised_mlp_f1(cfg, seed)
    return None if reference is None else float(cfg.fl.target_f1_frac) * reference


def run_scenario(cfg: Config, seed: int, scenario: Scenario, strategies: list[str], force: bool = False) -> dict[str, Any]:
    """Local-only, the fragmentation diagnostic and the federated strategies for one scenario.

    A saved result is reused only if it was produced from the same data and
    settings (see ``fl.data.run_fingerprint``); ``force`` recomputes regardless.
    """
    out: dict[str, Any] = {}
    root = cfg.path("results_dir", "fl", f"seed_{seed}")
    fingerprint = run_fingerprint(cfg, seed)
    for name, runner in [("local", run_local_only), ("pooled_local_features", run_pooled_local_features)]:
        saved = None if force else cached_result(root / f"{scenario.name}__{name}.json", fingerprint)
        out[name] = saved or runner(cfg, seed, scenario)
    for strategy in strategies:
        saved = None if force else cached_result(result_path(cfg, seed, scenario, strategy), fingerprint)
        out[strategy] = saved or run_federated(cfg, seed, scenario, strategy, target_f1=target_f1(cfg, seed))
    return out


def run_experiments(cfg: Config, seeds: list[int], force: bool = False) -> None:
    plan = experiment_plan(cfg)
    total = len(seeds) * len(plan)
    done = 0
    for seed in seeds:
        for scenario, strategies in plan:
            done += 1
            log.info("[%d/%d] seed %d  %s  %s", done, total, seed, scenario.label, strategies)
            run_scenario(cfg, seed, scenario, strategies, force)
