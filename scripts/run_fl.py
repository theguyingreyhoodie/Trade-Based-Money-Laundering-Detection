#!/usr/bin/env python
"""Phase 4: the primary federated run.

Trains the global model for the scenario in ``fl.primary`` (default: 3 clients,
non-IID Dirichlet alpha = 0.5, FedAvg) on ``project.seed``. Every client update
hash and every global model hash is registered on the evidence ledger, and the
global model is saved for risk scoring (Phase 6).
"""

from __future__ import annotations

from tbml.fl.experiments import primary_scenario, target_f1
from tbml.fl.simulation import run_federated
from tbml.utils import get_logger, parse_cli, save_json

log = get_logger("run_fl")


def _args(parser) -> None:
    parser.add_argument("--no-ledger", action="store_true", help="do not register model hashes on the evidence ledger")


def main() -> None:
    args, cfg = parse_cli(__doc__, _args)
    seed = int(cfg.project.seed)
    scenario = primary_scenario(cfg)
    strategy = str(cfg.fl.primary.strategy)
    sink, ledger = None, None
    if not args.no_ledger:
        from tbml.ledger.audit import ModelAuditSink
        from tbml.ledger.factory import open_ledger

        ledger = open_ledger(cfg)
        sink = ModelAuditSink(ledger, run_id=f"seed{seed}-{scenario.name}-{strategy}")
    result = run_federated(cfg, seed, scenario, strategy, audit_sink=sink, target_f1=target_f1(cfg, seed), tag="primary")
    if sink is not None:
        result["ledger"] = sink.summary()
        ledger.close()
    save_json(result, cfg.path("results_dir", "fl", "primary.json"))
    log.info("primary run saved to results/fl/primary.json")


if __name__ == "__main__":
    main()
