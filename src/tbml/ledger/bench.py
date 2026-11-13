"""Measurements of the evidence layer: latency, throughput, cost per record, storage growth."""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np

from tbml.config import Config
from tbml.ledger.base import EvidenceLedger
from tbml.ledger.merkle import merkle_root
from tbml.utils import get_logger

log = get_logger(__name__)


def _digests(tag: str, n: int) -> list[str]:
    return [hashlib.sha256(f"bench|{tag}|{i}".encode()).hexdigest() for i in range(n)]


def _percentiles(seconds: list[float]) -> dict[str, float]:
    ms = np.array(seconds) * 1000.0
    return {"p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)), "mean_ms": float(ms.mean()), "max_ms": float(ms.max()), "n": int(len(ms))}


def bench_single(ledger: EvidenceLedger, org: str, n: int, tag: str = "single") -> dict[str, Any]:
    """One ledger write per record."""
    latencies, costs = [], []
    start = time.perf_counter()
    for i, d in enumerate(_digests(tag, n)):
        receipt = ledger.register(d, "bench", f"{tag}-{i:06d}", org)
        latencies.append(receipt.latency_s)
        costs.append(receipt.cost)
    elapsed = time.perf_counter() - start
    return {"records": n, "latency": _percentiles(latencies), "records_per_second": n / elapsed, "cost_per_record": float(np.mean(costs)), "seconds": elapsed}


def bench_batches(ledger: EvidenceLedger, org: str, n_items: int, batch_sizes: list[int]) -> list[dict[str, Any]]:
    """Several records per ledger write."""
    rows = []
    for size in batch_sizes:
        digests = _digests(f"batch{size}", n_items)
        refs = [f"b{size}-{i:06d}" for i in range(n_items)]
        costs, n_writes = 0, 0
        start = time.perf_counter()
        for offset in range(0, n_items, size):
            receipts = ledger.register_batch(digests[offset : offset + size], "bench", refs[offset : offset + size], org)
            costs += sum(r.cost for r in receipts)
            n_writes += 1
        elapsed = time.perf_counter() - start
        rows.append({"batch_size": size, "records": n_items, "ledger_writes": n_writes, "records_per_second": n_items / elapsed,
                     "cost_per_record": costs / n_items, "ms_per_write": 1000.0 * elapsed / n_writes, "seconds": elapsed})
    return rows


def bench_merkle(ledger: EvidenceLedger, org: str, n_items: int, batch_size: int) -> dict[str, Any]:
    """One Merkle root per batch of records: the ledger stores one hash per batch."""
    digests = _digests("merkle", n_items)
    before = ledger.stats()["storage_bytes"]
    costs, n_writes = 0, 0
    start = time.perf_counter()
    for b, offset in enumerate(range(0, n_items, batch_size)):
        receipt = ledger.register(merkle_root(digests[offset : offset + batch_size]), "bench_batch", f"merkle-{b:05d}", org)
        costs += receipt.cost
        n_writes += 1
    elapsed = time.perf_counter() - start
    return {"batch_size": batch_size, "records": n_items, "ledger_writes": n_writes, "records_per_second": n_items / elapsed,
            "cost_per_record": costs / n_items, "storage_bytes_per_record": (ledger.stats()["storage_bytes"] - before) / n_items, "seconds": elapsed}


def bench_verify(ledger: EvidenceLedger, n: int) -> dict[str, Any]:
    """Read latency of ``verify`` for hashes that are on record."""
    latencies = []
    for d in _digests("single", n):
        start = time.perf_counter()
        ledger.verify(d)
        latencies.append(time.perf_counter() - start)
    return _percentiles(latencies)


def bench_storage_growth(ledger: EvidenceLedger, org: str, steps: int = 8, per_step: int = 50) -> list[dict[str, Any]]:
    """Ledger storage after each block of registrations (it grows linearly with the record count)."""
    rows = [{"records": ledger.stats()["n_records"], "storage_bytes": ledger.stats()["storage_bytes"]}]
    for s in range(steps):
        digests = _digests(f"growth{s}", per_step)
        ledger.register_batch(digests, "bench", [f"g{s}-{i:04d}" for i in range(per_step)], org)
        stats = ledger.stats()
        rows.append({"records": stats["n_records"], "storage_bytes": stats["storage_bytes"]})
    return rows


def bench_block_interval(ledger: Any, org: str, interval_s: float, n_sequential: int = 8, n_concurrent: int = 100) -> dict[str, Any]:
    """EVM only: latency and throughput when blocks are mined on a timer, as on a real permissioned chain."""
    ledger.set_mining(int(interval_s * 1000))
    try:
        latencies = [ledger.register(d, "bench", f"interval-{i:04d}", org).latency_s for i, d in enumerate(_digests("interval", n_sequential))]
        sender = ledger.organisations()[org]
        digests = _digests("interval-many", n_concurrent)
        start = time.perf_counter()
        txs = [ledger.contract.functions.registerEvidence(bytes.fromhex(d), "bench", f"im-{i:04d}", int(time.time())).transact({"from": sender})
               for i, d in enumerate(digests)]
        submitted = time.perf_counter() - start
        blocks = {int(ledger.w3.eth.wait_for_transaction_receipt(tx, poll_latency=0.01).blockNumber) for tx in txs}
        elapsed = time.perf_counter() - start
    finally:
        ledger.set_mining(None)
    return {"block_interval_s": interval_s, "sequential_latency": _percentiles(latencies), "concurrent": {
        "records": n_concurrent, "submit_seconds": submitted, "seconds_until_all_final": elapsed, "blocks_used": len(blocks),
        "records_per_second": n_concurrent / elapsed}}


def run_benchmark(ledger: EvidenceLedger, cfg: Config) -> dict[str, Any]:
    """Full benchmark on an empty ledger; returns a JSON-serialisable report."""
    b = cfg.ledger.benchmark
    org = "bench_org"
    ledger.register_organisation(org)
    sizes = [int(s) for s in b.batch_sizes if int(s) <= int(cfg.ledger.max_batch_items)]
    report: dict[str, Any] = {"backend": ledger.backend, "cost_unit": ledger.stats()["cost_unit"]}
    report["single"] = bench_single(ledger, org, int(b.n_single))
    report["verify"] = bench_verify(ledger, min(int(b.n_single), 200))
    report["batched"] = bench_batches(ledger, org, int(b.n_batched_items), sizes)
    report["merkle"] = bench_merkle(ledger, org, int(b.n_batched_items) * 5, int(cfg.ledger.doc_batch_size))
    report["storage_growth"] = bench_storage_growth(ledger, org)
    growth = report["storage_growth"]
    report["storage_bytes_per_record"] = (growth[-1]["storage_bytes"] - growth[0]["storage_bytes"]) / max(growth[-1]["records"] - growth[0]["records"], 1)
    if ledger.backend == "evm":
        report["block_interval"] = bench_block_interval(ledger, org, float(b.block_interval_s))
        report["deploy_gas"] = getattr(ledger, "deploy_gas", None)
    report["final_stats"] = ledger.stats()
    single = report["single"]
    log.info("%s ledger: single write p50 %.1f ms / p95 %.1f ms, %.0f records/s, %.0f %s per record", ledger.backend, single["latency"]["p50_ms"],
             single["latency"]["p95_ms"], single["records_per_second"], single["cost_per_record"], report["cost_unit"])
    return report
