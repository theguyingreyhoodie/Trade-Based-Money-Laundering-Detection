"""Flower ClientApp: one simulated bank.

Each simulated node is one client bank. Its `partition-id` (0..n-1) picks the folder
data/partitions/client_<id>/, and the bank only ever reads that folder.

mode "naive": the bank keeps only the core columns (shared by all banks) with the shared
category lists, so every bank has the same input layout, and the WHOLE BankModel is
exchanged and averaged.

mode "encoder_head" (our method): the bank uses its full own schema and its own category
lists. Only the SharedHead is exchanged and averaged. The encoder stays at the bank: it is
kept in the node's `context.state` between rounds and is never put in a reply message.

mode "hybrid": the bank's columns are laid out as [core | own]. A core encoder (on the core
columns, with the shared category lists) and the head are exchanged and averaged; the private
encoder on the bank's own columns stays in `context.state`.

mode "zero_filled" (baseline): all banks agree on one global column list (the union of their
columns) and lay out their data in it; columns a bank lacks are 0 with a 0/1 "<missing>" flag
(preprocess.load_client_zero_filled). The WHOLE BankModel is exchanged and averaged.

In every mode the rule is the same: a parameter whose name starts with PRIVATE_PREFIX[mode]
stays at the bank, everything else is sent to the server (see shared_state).

After the last round each bank writes its own test scores to <out_dir>/scores/. That file
is for the experimenter to compute results; it is not sent to the server.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from ..features import SHARED_VOCAB
from ..profiles import PROFILE_VOCAB
from ..metrics import best_f1_threshold
from ..metrics import evaluate as eval_metrics
from ..models import BankModel, HybridModel, pos_weight, predict_proba, train_epochs
from ..preprocess import ClientData, load_client, load_client_hybrid, load_client_zero_filled, undersample
from .strategy import AUDIT_KEY, hash_parameters

log = logging.getLogger(__name__)

MODES = ("naive", "zero_filled", "encoder_head", "hybrid")
# Parameters whose names start with this prefix never leave the bank (None: everything is shared).
PRIVATE_PREFIX = {"naive": None, "zero_filled": None, "encoder_head": "encoder.", "hybrid": "private_encoder."}
# Category lists for every categorical column, for the zero-filled layout.
ALL_VOCAB = {**SHARED_VOCAB, **PROFILE_VOCAB}
ENCODER_KEY = "encoder"        # context.state: this bank's private encoder weights
PROGRESS_KEY = "progress"      # context.state: {"encoder_rounds": n}


def new_model(mode: str, d: ClientData) -> torch.nn.Module:
    if mode == "hybrid":
        return HybridModel(d.core_dim, d.input_dim - d.core_dim)
    return BankModel(input_dim=d.input_dim)


def is_private(mode: str, key: str) -> bool:
    return PRIVATE_PREFIX[mode] is not None and key.startswith(PRIVATE_PREFIX[mode])


def shared_state(model: torch.nn.Module, mode: str) -> dict[str, torch.Tensor]:
    """The parameters this mode sends to the server."""
    return {k: v for k, v in model.state_dict().items() if not is_private(mode, k)}


def private_state(model: torch.nn.Module, mode: str) -> dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if is_private(mode, k)}


def make_client_app(partitions_dir: str | Path, out_dir: str | Path, mode: str = "naive",
                    lr: float = 1e-3, batch_size: int = 512, local_epochs: int = 1,
                    neg_per_pos: int = 20, seed: int = 42,
                    global_columns: list[str] | None = None) -> ClientApp:
    """global_columns: the agreed column list (zero_filled mode only)."""
    if mode not in MODES:
        raise ValueError(f"Unknown FL mode {mode!r}; expected one of {MODES}")
    if mode == "zero_filled" and not global_columns:
        raise ValueError("zero_filled mode needs global_columns")
    has_private = PRIVATE_PREFIX[mode] is not None
    app = ClientApp()
    # Per worker process, keep only the most recent bank's data: Ray does not pin a bank to a
    # worker, so caching every bank a worker has served would hold all partitions many times.
    cache: dict[int, ClientData] = {}

    def bank(context: Context) -> tuple[int, ClientData]:
        pid = int(context.node_config["partition-id"])
        if pid not in cache:
            cache.clear()
            path = Path(partitions_dir) / f"client_{pid}"
            if mode == "hybrid":
                cache[pid] = load_client_hybrid(path, SHARED_VOCAB)
            elif mode == "naive":
                cache[pid] = load_client(path, "core", SHARED_VOCAB)
            elif mode == "zero_filled":
                cache[pid] = load_client_zero_filled(path, global_columns, ALL_VOCAB)
            else:
                cache[pid] = load_client(path)
        return pid, cache[pid]

    def build_model(msg: Message, context: Context, pid: int, d: ClientData) -> torch.nn.Module:
        """Global weights from the server, plus this bank's own private encoder (if any)."""
        torch.set_num_threads(1)  # simulated banks run in parallel processes
        rnd = int(msg.content["config"]["server-round"])
        torch.manual_seed(seed + 1000 * pid + rnd)  # reproducible init and shuffling per bank/round
        model = new_model(mode, d)
        received = msg.content["arrays"].to_torch_state_dict()
        if set(received) != set(shared_state(model, mode)):
            raise ValueError(f"Server sent {sorted(received)}, expected the {mode} shared parameters")
        state = model.state_dict()
        state.update(received)
        if ENCODER_KEY in context.state:  # first round: freshly initialised encoder
            state.update(context.state[ENCODER_KEY].to_torch_state_dict())
        model.load_state_dict(state)
        return model

    def encoder_rounds(context: Context) -> int:
        return int(context.state[PROGRESS_KEY]["encoder_rounds"]) if PROGRESS_KEY in context.state else 0

    @app.train()
    def train(msg: Message, context: Context) -> Message:
        pid, d = bank(context)
        model = build_model(msg, context, pid, d)
        rnd = int(msg.content["config"]["server-round"])
        X, y = undersample(d.X["train"], d.y["train"], neg_per_pos=neg_per_pos, seed=seed + rnd)
        loss = train_epochs(model, X, y, epochs=local_epochs, lr=lr, batch_size=batch_size,
                            weight=pos_weight(y))
        metrics = {"train_loss": float(loss), "num-examples": len(d.y["train"])}
        shared = ArrayRecord(shared_state(model, mode))            # the ONLY thing that leaves the bank
        if has_private:
            context.state[ENCODER_KEY] = ArrayRecord(private_state(model, mode))
            context.state[PROGRESS_KEY] = ConfigRecord({"encoder_rounds": encoder_rounds(context) + 1})
            metrics["encoder_rounds"] = encoder_rounds(context)   # should equal the round number
        # The bank declares the hash of what it sends; the server checks it and logs it for the audit.
        audit = ConfigRecord({"bank": d.name, "update-hash": hash_parameters(shared)})
        return Message(content=RecordDict({"arrays": shared, "metrics": MetricRecord(metrics), AUDIT_KEY: audit}),
                       reply_to=msg)

    @app.evaluate()
    def evaluate(msg: Message, context: Context) -> Message:
        pid, d = bank(context)
        model = build_model(msg, context, pid, d)
        val = predict_proba(model, d.X["val"])
        has_pos = d.y["val"].sum() > 0
        thr = best_f1_threshold(d.y["val"], val) if has_pos else 0.5
        m = eval_metrics(d.y["val"], val, thr)
        metrics = MetricRecord({
            "val_pr_auc": float(np.nan_to_num(m["pr_auc"])),  # 0 when the bank has no val positives
            "val_f1": m["f1"],
            "val_positives": m["positives"],
            "num-examples": len(d.y["val"]),
        })
        cfg = msg.content["config"]
        if int(cfg["server-round"]) == int(cfg["num-rounds"]):
            out = Path(out_dir) / "scores"
            out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"txn_id": d.txn_ids["test"], "label": d.y["test"].astype(int),
                          "score": predict_proba(model, d.X["test"]), "threshold": thr}
                         ).to_parquet(out / f"client_{pid}.parquet", index=False)
        return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)

    return app
