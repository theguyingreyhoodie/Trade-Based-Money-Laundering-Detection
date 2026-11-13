"""Flower server side: FedAvg / FedProx with auditing, and the round loop.

The strategies are Flower's own ``FedAvg`` and ``FedProx`` (message API),
extended by a mixin that, for every client reply,

1. validates it against the client-boundary whitelist (``fl.boundary``),
2. accounts the bytes sent and received,
3. recomputes the SHA-256 of the received update and compares it with the hash
   the client reported, then records it in the audit trail,
4. optionally clips the update norm (poisoning defence) or unmasks a securely
   aggregated sum.

The weighted average itself is Flower's ``aggregate_arrayrecords``.

Evaluation is federated as well: clients return score histograms, the server
adds them up, picks the decision threshold on the summed validation histogram
and reads all metrics off the summed test histogram.
"""

from __future__ import annotations

import random
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MessageType, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, FedProx
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords

from tbml.config import Config
from tbml.eval.metrics import ScoreHist, evaluate_hists
from tbml.fl.boundary import EVALUATE, STATS, TRAIN, BoundarySpec, check_reply, message_bytes
from tbml.ledger.hashing import hash_arrays
from tbml.models.mlp import moments_to_scaler
from tbml.utils import get_logger

log = get_logger(__name__)

AuditSink = Callable[[dict[str, Any]], None]


@dataclass
class RunState:
    """Everything the server learns and measures during one federated run."""

    cfg: Config
    spec: BoundarySpec
    names: list[str]
    trainable: list[bool]
    n_clients: int
    audit_sink: AuditSink | None = None
    norm_clip: float = 0.0
    secagg: dict[str, Any] | None = None
    comm: list[dict[str, Any]] = field(default_factory=list)
    audit: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    node_to_client: dict[int, int] = field(default_factory=dict)
    global_weights: list[np.ndarray] | None = None
    client_hists: dict[int, tuple[ScoreHist, ScoreHist]] = field(default_factory=dict)
    update_norms: dict[int, float] = field(default_factory=dict)
    clipped: int = 0

    def record_bytes(self, server_round: int, phase: str, direction: str, n_bytes: int) -> None:
        self.comm.append({"round": server_round, "phase": phase, "direction": direction, "bytes": int(n_bytes)})

    def record_audit(self, entry: dict[str, Any]) -> None:
        self.audit.append(entry)
        if self.audit_sink is not None:
            self.audit_sink(entry)


def to_record(names: list[str], weights: Iterable[np.ndarray]) -> ArrayRecord:
    return ArrayRecord(array_dict=OrderedDict((name, Array(np.asarray(w))) for name, w in zip(names, weights)))


def from_record(record: ArrayRecord, names: list[str]) -> list[np.ndarray]:
    return [record[name].numpy() for name in names]


def _valid(replies: Iterable[Message], what: str) -> list[Message]:
    replies = list(replies)
    errors = [m for m in replies if m.has_error()]
    if errors:
        reasons = "; ".join(str(m.error.reason)[:300] for m in errors[:3])
        raise RuntimeError(f"{len(errors)} client(s) failed during {what}: {reasons}")
    if not replies:
        raise RuntimeError(f"no replies received during {what}")
    return replies


class AuditedAggregation:
    """Mixin for Flower strategies: boundary check, byte accounting, hashes, clipping."""

    state: RunState

    def attach(self, state: RunState) -> "AuditedAggregation":
        self.state = state
        return self

    # ------------------------------------------------------------------ training
    def configure_train(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid) -> Iterable[Message]:
        messages = list(super().configure_train(server_round, arrays, config, grid))  # type: ignore[misc]
        self.state.global_weights = from_record(arrays, self.state.names)
        for msg in messages:
            self.state.record_bytes(server_round, "train", "down", message_bytes(msg.content))
        return messages

    def aggregate_train(self, server_round: int, replies: Iterable[Message]) -> tuple[ArrayRecord | None, MetricRecord | None]:
        st = self.state
        contents: list[tuple[int, RecordDict]] = []
        for msg in _valid(replies, f"training round {server_round}"):
            content = msg.content
            receipt = check_reply(content, st.spec, TRAIN)
            st.record_bytes(server_round, "train", "up", receipt["bytes"])
            metrics = content["metrics"]
            client = int(metrics["partition-id"])
            record = content["arrays"]
            digest = hash_arrays((name, array.numpy()) for name, array in record.items())
            claimed = str(content["meta"]["update-sha256"])
            if digest != claimed:
                raise RuntimeError(f"round {server_round}: update of client {client} does not match its reported hash")
            st.record_audit({"round": server_round, "kind": "client_update", "client": client, "sha256": digest,
                             "n_examples": int(metrics["num-examples"]), "bytes": receipt["bytes"]})
            contents.append((client, content))
        contents.sort(key=lambda item: item[0])  # fixed order -> reproducible floating-point sums

        if st.secagg:
            from tbml.fl.extensions import unmask_sum

            weights = unmask_sum([c["arrays"] for _, c in contents], st.names, st.secagg)
            aggregated = to_record(st.names, weights)
        else:
            if st.norm_clip > 0:
                contents = [(client, self._clip(client, content)) for client, content in contents]
            averaged = aggregate_arrayrecords([c for _, c in contents], self.weighted_by_key)  # type: ignore[attr-defined]
            assert st.global_weights is not None
            # trained arrays come from the weighted average; buffers (the shared scaler) are kept as sent
            merged = [averaged[name].numpy().astype(np.float32) if t else g for name, t, g in zip(st.names, st.trainable, st.global_weights)]
            aggregated = to_record(st.names, merged)

        total = sum(float(c["metrics"]["num-examples"]) for _, c in contents)
        loss = sum(float(c["metrics"]["train-loss"]) * float(c["metrics"]["num-examples"]) for _, c in contents) / max(total, 1.0)
        global_digest = hash_arrays((name, array.numpy()) for name, array in aggregated.items())
        st.record_audit({"round": server_round, "kind": "global_model", "client": None, "sha256": global_digest,
                         "n_examples": int(total), "bytes": message_bytes(RecordDict({"arrays": aggregated}))})
        return aggregated, MetricRecord({"train-loss": float(loss), "n-clients": len(contents)})

    def _clip(self, client: int, content: RecordDict) -> RecordDict:
        """Scale a client's update so that its L2 norm is at most ``norm_clip``."""
        st = self.state
        assert st.global_weights is not None
        local = from_record(content["arrays"], st.names)
        delta = [l.astype(np.float64) - g.astype(np.float64) for l, g in zip(local, st.global_weights)]
        norm = float(np.sqrt(sum(float((d**2).sum()) for d, t in zip(delta, st.trainable) if t)))
        st.update_norms[client] = norm
        if norm <= st.norm_clip:
            return content
        st.clipped += 1
        scale = st.norm_clip / norm
        clipped = [(g + d * scale).astype(np.float32) if t else l for l, g, d, t in zip(local, st.global_weights, delta, st.trainable)]
        return RecordDict({"arrays": to_record(st.names, clipped), "metrics": content["metrics"]})

    # ------------------------------------------------------------------ evaluation
    def configure_evaluate(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid) -> Iterable[Message]:
        messages = list(super().configure_evaluate(server_round, arrays, config, grid))  # type: ignore[misc]
        for msg in messages:
            self.state.record_bytes(server_round, "evaluate", "down", message_bytes(msg.content))
        return messages

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]) -> MetricRecord | None:
        st = self.state
        n_bins = int(st.cfg.fl.hist_bins)
        val, test = ScoreHist.empty(n_bins), ScoreHist.empty(n_bins)
        weighted_loss, total = 0.0, 0.0
        st.client_hists = {}
        for msg in _valid(replies, f"evaluation round {server_round}"):
            receipt = check_reply(msg.content, st.spec, EVALUATE)
            st.record_bytes(server_round, "evaluate", "up", receipt["bytes"])
            metrics = msg.content["metrics"]
            h_val = ScoreHist.from_list(metrics["hist-val"], n_bins)
            h_test = ScoreHist.from_list(metrics["hist-test"], n_bins)
            st.client_hists[int(metrics["partition-id"])] = (h_val, h_test)
            val, test = val + h_val, test + h_test
            weighted_loss += float(metrics["val-loss"]) * float(metrics["num-examples"])
            total += float(metrics["num-examples"])
        return MetricRecord({"val-loss": weighted_loss / max(total, 1.0), "hist-val": val.to_list(), "hist-test": test.to_list()})


class AuditedFedAvg(AuditedAggregation, FedAvg):
    """Flower FedAvg with the auditing mixin."""


class AuditedFedProx(AuditedAggregation, FedProx):
    """Flower FedProx (sends ``proximal-mu`` to the clients) with the auditing mixin."""


def make_strategy(cfg: Config, name: str, n_clients: int, state: RunState) -> AuditedAggregation:
    """FedAvg or FedProx configured from ``cfg.fl``."""
    f = cfg.fl
    n_train = max(2, int(round(n_clients * float(f.fraction_train))))
    common = dict(
        fraction_train=float(f.fraction_train),
        fraction_evaluate=float(f.fraction_evaluate),
        min_train_nodes=min(n_clients, n_train),
        min_evaluate_nodes=n_clients if float(f.fraction_evaluate) >= 1.0 else 2,
        min_available_nodes=n_clients,
    )
    if name == "fedavg":
        return AuditedFedAvg(**common).attach(state)
    if name == "fedprox":
        return AuditedFedProx(proximal_mu=float(f.proximal_mu), **common).attach(state)
    raise ValueError(f"unknown strategy '{name}' (fedavg | fedprox)")


def wait_for_nodes(grid: Grid, n: int, timeout: float = 120.0) -> list[int]:
    deadline = time.monotonic() + timeout
    while True:
        nodes = list(grid.get_node_ids())
        if len(nodes) >= n:
            return nodes
        if time.monotonic() > deadline:
            raise TimeoutError(f"only {len(nodes)} of {n} client nodes connected")
        time.sleep(0.05)


def federated_scaler(grid: Grid, state: RunState, timeout: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Round 0: feature mean / std from per-client sums (no rows are shared)."""
    nodes = wait_for_nodes(grid, state.n_clients)
    request = RecordDict({"config": ConfigRecord({"phase": "feature-stats"})})
    messages = [Message(content=request, message_type=MessageType.QUERY, dst_node_id=node) for node in nodes]
    for msg in messages:
        state.record_bytes(0, "stats", "down", message_bytes(msg.content))
    n_total, total, total_sq = 0.0, None, None
    for msg in _valid(grid.send_and_receive(messages, timeout=timeout), "feature statistics"):
        receipt = check_reply(msg.content, state.spec, STATS)
        state.record_bytes(0, "stats", "up", receipt["bytes"])
        metrics = msg.content["metrics"]
        state.node_to_client[int(msg.metadata.src_node_id)] = int(metrics["partition-id"])
        s = np.asarray(metrics["feat-sum"], dtype=np.float64)
        q = np.asarray(metrics["feat-sumsq"], dtype=np.float64)
        n_total += float(metrics["num-examples"])
        total = s if total is None else total + s
        total_sq = q if total_sq is None else total_sq + q
    mean, std = moments_to_scaler(n_total, total, total_sq)
    return mean, std, int(n_total)


def run_rounds(
    grid: Grid,
    strategy: AuditedAggregation,
    state: RunState,
    initial_weights: list[np.ndarray],
    rounds: int,
    train_config: dict[str, Any],
    seed: int,
    timeout: float = 600.0,
    on_round: Callable[[int, dict[str, Any], list[np.ndarray]], None] | None = None,
) -> dict[str, Any]:
    """Federated standardisation, then ``rounds`` rounds of train + federated evaluation.

    Returns the per-round history and the best global model (highest validation
    PR-AUC, the same model-selection rule as the centralised MLP).
    """
    random.seed(seed)  # Flower samples nodes with the global ``random`` module
    metric = str(state.cfg.models.threshold_metric)
    start = time.perf_counter()

    mean, std, n_train = federated_scaler(grid, state, timeout)
    weights = [w.copy() for w in initial_weights]
    weights[state.names.index("scaler.mean")] = mean
    weights[state.names.index("scaler.std")] = std
    arrays = to_record(state.names, weights)

    best: dict[str, Any] = {"round": 0, "val_pr_auc": -1.0}
    cumulative = sum(row["bytes"] for row in state.comm)
    for server_round in range(1, rounds + 1):
        t0 = time.perf_counter()
        before = len(state.comm)
        round_config = dict(train_config)
        round_config["lr"] = float(train_config["lr"]) * float(state.cfg.fl.lr_decay) ** (server_round - 1)
        config = ConfigRecord(round_config)
        replies = grid.send_and_receive(strategy.configure_train(server_round, arrays, config, grid), timeout=timeout)
        arrays, train_metrics = strategy.aggregate_train(server_round, replies)
        replies = grid.send_and_receive(strategy.configure_evaluate(server_round, arrays, ConfigRecord({}), grid), timeout=timeout)
        eval_metrics = strategy.aggregate_evaluate(server_round, replies)

        n_bins = int(state.cfg.fl.hist_bins)
        h_val = ScoreHist.from_list(eval_metrics["hist-val"], n_bins)
        h_test = ScoreHist.from_list(eval_metrics["hist-test"], n_bins)
        result = evaluate_hists(h_val, h_test, metric)
        traffic = {(row["phase"], row["direction"]): 0 for row in state.comm[before:]}
        for row in state.comm[before:]:
            traffic[(row["phase"], row["direction"])] += row["bytes"]
        round_bytes = sum(traffic.values())
        cumulative += round_bytes
        row = {
            "round": server_round,
            "train_loss": float(train_metrics["train-loss"]),
            "val_loss": float(eval_metrics["val-loss"]),
            "val_f1": result["val"]["f1"],
            "val_pr_auc": result["val"]["pr_auc"],
            "test_f1": result["f1"],
            "test_precision": result["precision"],
            "test_recall": result["recall"],
            "test_pr_auc": result["pr_auc"],
            "test_roc_auc": result["roc_auc"],
            "threshold": result["threshold"],
            "bytes_train_down": traffic.get(("train", "down"), 0),
            "bytes_train_up": traffic.get(("train", "up"), 0),
            "bytes_eval_down": traffic.get(("evaluate", "down"), 0),
            "bytes_eval_up": traffic.get(("evaluate", "up"), 0),
            "bytes_round": round_bytes,
            "bytes_cumulative": cumulative,
            "seconds": time.perf_counter() - t0,
        }
        state.history.append(row)
        current = from_record(arrays, state.names)
        if result["val"]["pr_auc"] > best["val_pr_auc"]:
            best = {
                "round": server_round,
                "val_pr_auc": result["val"]["pr_auc"],
                "weights": [w.copy() for w in current],
                "metrics": result,
                "client_hists": dict(state.client_hists),
                "hist_val": h_val,
                "hist_test": h_test,
            }
        if on_round is not None:
            on_round(server_round, row, current)
        log.info("round %2d/%d  loss %.4f  val PR-AUC %.3f  test F1 %.3f  PR-AUC %.3f  (%.0f kB, %.1fs)", server_round, rounds,
                 row["train_loss"], row["val_pr_auc"], row["test_f1"], row["test_pr_auc"], round_bytes / 1e3, row["seconds"])

    last = evaluate_hists(ScoreHist.from_list(eval_metrics["hist-val"], n_bins), ScoreHist.from_list(eval_metrics["hist-test"], n_bins), metric)
    return {"best": best, "last": last, "n_train_examples": n_train, "seconds": time.perf_counter() - start,
            "scaler": {"mean": mean, "std": std}}
