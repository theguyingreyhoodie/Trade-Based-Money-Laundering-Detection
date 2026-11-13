#!/usr/bin/env python
"""Phase 3: baselines. Rules engine plus centralised logistic regression,
random forest, LightGBM and MLP on the pooled data.

Writes ``results/baselines/seed_<s>.json`` and the fitted models and scores to
``artifacts/seed_<s>/baselines/``.
"""

from __future__ import annotations

from tbml.features.pipeline import load_central_features
from tbml.models.baselines import run_and_save
from tbml.rules.engine import rule_book_markdown
from tbml.utils import get_logger, parse_cli, seed_everything, timed

log = get_logger("run_baselines")


def _args(parser) -> None:
    parser.add_argument("--all-seeds", action="store_true", help="run every seed in project.seeds")
    parser.add_argument("--docs-only", action="store_true", help="only regenerate docs/rules.md")


def main() -> None:
    args, cfg = parse_cli(__doc__, _args)
    doc = cfg.root / "docs" / "rules.md"
    doc.write_text(rule_book_markdown(cfg), encoding="utf-8")
    log.info("wrote %s", doc)
    if args.docs_only:
        return
    seeds = list(cfg.project.seeds) if args.all_seeds else [cfg.project.seed]
    for seed in seeds:
        seed_everything(seed)
        fs = load_central_features(cfg, seed)
        with timed(f"baselines seed {seed}", log):
            run_and_save(cfg, fs, seed)


if __name__ == "__main__":
    main()
