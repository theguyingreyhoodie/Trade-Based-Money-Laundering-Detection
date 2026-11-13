#!/usr/bin/env python
"""Phase 4: the federated experiment sweep.

For every seed in ``project.seeds``: IID, Dirichlet non-IID (every alpha) and
quantity-skew splits; 3, 5 and 10 clients; FedAvg, FedProx on non-IID splits,
local-only training and the pooled-local-features diagnostic.

Results go to ``results/fl/seed_<s>/<scenario>__<method>.json``. Finished runs
are reused unless ``--force`` is given.
"""

from __future__ import annotations

from tbml.fl.experiments import experiment_plan, run_experiments
from tbml.utils import get_logger, parse_cli, timed

log = get_logger("run_fl_experiments")


def _args(parser) -> None:
    parser.add_argument("--force", action="store_true", help="re-run experiments whose result file already exists")
    parser.add_argument("--list", action="store_true", help="print the plan and exit")


def main() -> None:
    args, cfg = parse_cli(__doc__, _args)
    plan = experiment_plan(cfg)
    if args.list:
        for scenario, strategies in plan:
            print(f"{scenario.name:12s} {scenario.label:42s} local, pooled_local_features, {', '.join(strategies)}")
        return
    with timed("federated experiment sweep", log):
        run_experiments(cfg, [int(s) for s in cfg.project.seeds], force=args.force)


if __name__ == "__main__":
    main()
