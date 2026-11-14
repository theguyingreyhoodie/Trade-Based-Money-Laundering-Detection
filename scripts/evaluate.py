#!/usr/bin/env python
"""Phase 7: evaluation and audit report.

Reads every result file, aggregates over seeds, writes the tables to
``results/tables/``, the figures to ``results/plots/`` and the report to
``results/RESULTS.md``. Sections whose results are missing are marked as such,
so the script can be run at any point of the pipeline.
"""

from __future__ import annotations

from tbml.eval.results_md import build_results
from tbml.utils import get_logger, parse_cli

log = get_logger("evaluate")


def main() -> None:
    _, cfg = parse_cli(__doc__)
    build_results(cfg)


if __name__ == "__main__":
    main()
