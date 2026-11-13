#!/usr/bin/env python
"""Phase 2: clean and link the raw records, then build the risk indicators.

Writes ``artifacts/seed_<s>/features/central.npz`` (pooled view, used by the
centralised baselines), a report in ``results/features/`` and the feature
dictionary ``docs/features.md``.
"""

from __future__ import annotations

from tbml.features.dictionary import FEATURES, features_markdown
from tbml.features.pipeline import build_central_features, feature_report
from tbml.utils import get_logger, parse_cli, save_json, timed

log = get_logger("build_features")


def _args(parser) -> None:
    parser.add_argument("--all-seeds", action="store_true", help="build for every seed in project.seeds")
    parser.add_argument("--docs-only", action="store_true", help="only regenerate docs/features.md")


def main() -> None:
    args, cfg = parse_cli(__doc__, _args)
    doc = cfg.root / "docs" / "features.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(features_markdown(), encoding="utf-8")
    log.info("wrote %s (%d features)", doc, len(FEATURES))
    if args.docs_only:
        return
    seeds = list(cfg.project.seeds) if args.all_seeds else [cfg.project.seed]
    for seed in seeds:
        with timed(f"features seed {seed}", log) as t:
            fs = build_central_features(cfg, seed)
        report = feature_report(fs, cfg)
        report["seconds"] = t["seconds"]
        save_json(report, cfg.path("results_dir", "features", f"report_seed_{seed}.json"))
        split = {k: (v["n"], round(v["illicit_rate"], 4)) for k, v in report["split"].items()}
        log.info("seed %d: %d trades x %d features, split %s", seed, len(fs), fs.X.shape[1], split)
        for typology, info in report["typologies"].items():
            best = info["top_features"][0]
            log.info("  %-20s best indicator %-34s separation %.3f", typology, best["feature"], best["separation"])


if __name__ == "__main__":
    main()
