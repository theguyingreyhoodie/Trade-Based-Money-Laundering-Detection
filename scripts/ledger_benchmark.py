#!/usr/bin/env python
"""Phase 5c: benchmark the evidence ledger.

Measures registration latency (p50 / p95), throughput with and without
batching, cost per record (gas on the EVM backend), read latency and storage
growth. On the EVM backend it also repeats the measurement with blocks mined on
a timer, which is how a real permissioned chain behaves.

The benchmark runs on its own, empty registry (a separate contract), so it does
not disturb the evidence registered by the pipeline. The pure-Python backend is
always measured as well, for comparison. Writes ``results/ledger/benchmark.json``.
"""

from __future__ import annotations

from tbml.ledger.bench import run_benchmark
from tbml.ledger.factory import ledger_dir, open_ledger, resolve_backend
from tbml.utils import get_logger, parse_cli, save_json

log = get_logger("ledger_benchmark")


def main() -> None:
    _, cfg = parse_cli(__doc__)
    report = {}
    backends = ["python"] + (["evm"] if resolve_backend(cfg) == "evm" else [])
    for backend in backends:
        ledger = open_ledger(cfg, backend=backend, fresh=True, state_dir=ledger_dir(cfg) / "bench")
        report[backend] = run_benchmark(ledger, cfg)
        ledger.close()
    if "evm" not in report:
        log.warning("EVM node not available: only the pure-Python ledger was benchmarked")
    save_json(report, cfg.path("results_dir", "ledger", "benchmark.json"))


if __name__ == "__main__":
    main()
