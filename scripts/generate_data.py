#!/usr/bin/env python
"""Phase 1: generate the synthetic trade dataset(s).

Writes, per seed, ``data/seed_<s>/`` with the raw per-owner tables
(``raw/bank``, ``raw/trade``, ``raw/customs``), the public reference tables,
the ground truth, and a summary in ``results/data/summary_seed_<s>.json``.
"""

from __future__ import annotations

from tbml.data.generator import generate_dataset, write_dataset
from tbml.data.schema import validate_dataset
from tbml.features.link import write_joined_ground_truth
from tbml.utils import get_logger, parse_cli, save_json, timed

log = get_logger("generate_data")


def main() -> None:
    args, cfg = parse_cli(__doc__, lambda p: p.add_argument("--all-seeds", action="store_true", help="generate every seed in project.seeds"))
    seeds = list(cfg.project.seeds) if args.all_seeds else [cfg.project.seed]
    for seed in seeds:
        with timed(f"generate seed {seed}", log) as t:
            ds = generate_dataset(cfg, seed)
        report = validate_dataset(ds, cfg)
        out = write_dataset(ds, cfg.seed_dir("data_dir", seed))
        joined = write_joined_ground_truth(ds, out)
        summary = {**{k: v for k, v in ds.meta.items() if k != "data_config"}, "validation": report, "seconds": t["seconds"], "joined_rows": int(len(joined))}
        save_json(summary, cfg.path("results_dir", "data", f"summary_seed_{seed}.json"))
        log.info("seed %d written to %s (%d checks passed)", seed, out, len(report["checks"]))


if __name__ == "__main__":
    main()
