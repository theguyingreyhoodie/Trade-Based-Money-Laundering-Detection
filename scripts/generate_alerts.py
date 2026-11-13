#!/usr/bin/env python
"""Phase 6: calibrated risk scores and explainable alerts.

Uses the global model of the primary federated run. Every institution scores
its own test-period trades (model + rules), the scores are calibrated on
validation histograms, and an alert is raised from the Medium band upwards.
Each alert gets SHAP explanations in plain English, the fired rules, a
suspected typology, and is registered on the evidence ledger.

Outputs in ``results/alerts/``: ``alerts.json``, ``report.md`` (readable sample),
``summary.json`` (bands, calibration, blend weight sensitivity, attribution).
"""

from __future__ import annotations

from tbml.data.generator import load_dataset
from tbml.explain.alerts import run_alert_pipeline
from tbml.features.link import link_dataset
from tbml.fl.data import load_client, scenario_dir
from tbml.fl.experiments import primary_scenario
from tbml.fl.simulation import load_global_model
from tbml.utils import get_logger, load_json, parse_cli, save_json

log = get_logger("generate_alerts")


def _args(parser) -> None:
    parser.add_argument("--no-ledger", action="store_true", help="do not register alerts on the evidence ledger")


def main() -> None:
    args, cfg = parse_cli(__doc__, _args)
    seed = int(cfg.project.seed)
    scenario = primary_scenario(cfg)
    strategy = str(cfg.fl.primary.strategy)
    model, meta = load_global_model(cfg, seed, scenario, strategy, tag="primary")
    primary = load_json(cfg.path("results_dir", "fl", "primary.json"))
    entry = next(a for a in primary["model_hashes"] if a["kind"] == "global_model" and a["round"] == meta["best_round"])
    model_info = {"run_id": f"seed{seed}-{scenario.name}-{strategy}", "round": int(meta["best_round"]), "global_model_sha256": entry["sha256"],
                  "global_model_tx": entry.get("tx_id", "")}
    root = scenario_dir(cfg, seed, scenario)
    clients = [load_client(root, c) for c in range(scenario.n_clients)]
    linked = link_dataset(load_dataset(cfg.seed_dir("data_dir", seed)))

    ledger = None
    if not args.no_ledger:
        from tbml.ledger.factory import open_ledger

        ledger = open_ledger(cfg)
    out = run_alert_pipeline(cfg, seed, model, model_info, clients, linked, ledger)
    if ledger is not None:
        ledger.close()

    results = cfg.path("results_dir", "alerts", create=True)
    save_json({"seed": seed, "model": model_info, "n_alerts": len(out["alerts"]), "alerts": out["alerts"]}, results / "alerts.json")
    save_json(out["summary"], results / "summary.json")
    (results / "report.md").write_text(out["report"], encoding="utf-8")
    artifacts = cfg.seed_dir("artifacts_dir", seed, "alerts", create=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    out["scores"].to_csv(artifacts / "scores.csv", index=False)
    save_json(out["truth"], artifacts / "truth.json")
    log.info("wrote %s (%d alerts), report.md and summary.json", results / "alerts.json", len(out["alerts"]))


if __name__ == "__main__":
    main()
