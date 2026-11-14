#!/usr/bin/env python
"""Stretch experiments (run after the core pipeline).

    secagg        FedAvg with pairwise additive masking: same model, server sees only masked updates
    dp            DP-SGD clients (Opacus) at several epsilon values: privacy / utility curve
    poison        one malicious client out of three, with and without median norm clipping
    personalised  local fine-tuning of the FedAvg model on every scenario of the sweep
    vertical      split neural network for the Bank / Trade registry / Customs picture

Each part writes ``results/stretch/<part>.json``. ``--only a,b`` runs a subset.
"""

from __future__ import annotations

from tbml.fl import stretch
from tbml.utils import get_logger, parse_cli, save_json, timed

log = get_logger("run_stretch")
PARTS = ["secagg", "dp", "poison", "personalised", "vertical"]


def main() -> None:
    args, cfg = parse_cli(__doc__, lambda p: p.add_argument("--only", default=",".join(PARTS), help="comma-separated subset of: " + ", ".join(PARTS)))
    chosen = [part.strip() for part in args.only.split(",") if part.strip()]
    unknown = sorted(set(chosen) - set(PARTS))
    if unknown:
        raise SystemExit(f"unknown part(s) {unknown}; choose from {PARTS}")
    seeds = [int(s) for s in cfg.project.seeds]
    runners = {
        "secagg": lambda: stretch.secure_aggregation(cfg, int(cfg.project.seed)),
        "dp": lambda: stretch.differential_privacy(cfg, seeds),
        "poison": lambda: stretch.poisoning(cfg, seeds),
        "personalised": lambda: stretch.personalisation(cfg, seeds),
        "vertical": lambda: stretch.vertical(cfg, seeds),
    }
    for part in chosen:
        with timed(f"stretch: {part}", log):
            save_json(runners[part](), cfg.path("results_dir", "stretch", f"{part}.json"))


if __name__ == "__main__":
    main()
