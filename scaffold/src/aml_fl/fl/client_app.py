"""Flower ClientApp: one simulated bank.

Each simulated node is one client bank. Its `partition-id` (0..n-1) picks the folder
data/partitions/client_<id>/, and the bank only ever reads that folder.

mode "naive": the bank keeps only the core columns (shared by all banks) with the shared
category lists, so every bank has the same input layout, and the WHOLE BankModel is
exchanged and averaged.

After the last round each bank writes its own test scores to <out_dir>/scores/. That file
is for the experimenter to compute results; it is not sent to the server.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from ..features import SHARED_VOCAB
from ..metrics import best_f1_threshold
from ..metrics import evaluate as eval_metrics
from ..models import BankModel, pos_weight, predict_proba, train_epochs
from ..preprocess import ClientData, load_client, undersample

log = logging.getLogger(__name__)

MODES = ("naive",)


def make_client_app(partitions_dir: str | Path, out_dir: str | Path, mode: str = "naive",
                    lr: float = 1e-3, batch_size: int = 512, local_epochs: int = 1,
                    neg_per_pos: int = 20, seed: int = 42) -> ClientApp:
    if mode not in MODES:
        raise ValueError(f"Unknown FL mode {mode!r}; expected one of {MODES}")
    app = ClientApp()
    cache: dict[int, ClientData] = {}  # per worker process: load each bank's data once

    def bank(context: Context) -> tuple[int, ClientData]:
        pid = int(context.node_config["partition-id"])
        if pid not in cache:
            cache[pid] = load_client(Path(partitions_dir) / f"client_{pid}", core_only=True,
                                     fixed_categories=SHARED_VOCAB)
        return pid, cache[pid]

    def global_model(msg: Message, d: ClientData) -> BankModel:
        torch.set_num_threads(1)  # simulated banks run in parallel processes
        model = BankModel(input_dim=d.input_dim)
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
        return model

    @app.train()
    def train(msg: Message, context: Context) -> Message:
        _, d = bank(context)
        model = global_model(msg, d)
        rnd = int(msg.content["config"]["server-round"])
        X, y = undersample(d.X["train"], d.y["train"], neg_per_pos=neg_per_pos, seed=seed + rnd)
        loss = train_epochs(model, X, y, epochs=local_epochs, lr=lr, batch_size=batch_size,
                            weight=pos_weight(y))
        metrics = MetricRecord({"train_loss": float(loss), "num-examples": len(d.y["train"])})
        return Message(content=RecordDict({"arrays": ArrayRecord(model.state_dict()), "metrics": metrics}),
                       reply_to=msg)

    @app.evaluate()
    def evaluate(msg: Message, context: Context) -> Message:
        pid, d = bank(context)
        model = global_model(msg, d)
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
