"""Flower ClientApp: what runs inside each participating institution.

A client loads only its own feature file (selected by the ``partition-id`` the
Flower runtime assigns to the node) and answers three kinds of messages:

``query``     per-feature count / sum / sum of squares of its training rows,
              so the federation can standardise features without sharing rows;
``train``     local training starting from the global model; returns the
              updated model, the number of training rows and the update hash;
``evaluate``  score histograms of the global model on its own validation and
              test rows (aggregates, not predictions).

Nothing else leaves the client; the server enforces that with
``tbml.fl.boundary``.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from tbml.config import Config
from tbml.eval.metrics import ScoreHist
from tbml.features.build import FeatureSet
from tbml.fl.data import load_client
from tbml.ledger.hashing import hash_arrays
from tbml.models.mlp import feature_moments, get_weights, make_loss, make_model, predict_proba, set_weights, train_epochs, weight_spec
from tbml.utils import derive_seed

_DATA_CACHE: dict[tuple[str, int], FeatureSet] = {}


def client_data(data_dir: str, partition_id: int) -> FeatureSet:
    """This client's own data (cached for the lifetime of the worker process)."""
    key = (str(data_dir), int(partition_id))
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = load_client(data_dir, partition_id)
    return _DATA_CACHE[key]


def to_array_record(names: list[str], weights: list[np.ndarray]) -> ArrayRecord:
    return ArrayRecord(array_dict=OrderedDict((name, Array(np.asarray(w))) for name, w in zip(names, weights)))


def from_array_record(record: ArrayRecord, names: list[str]) -> list[np.ndarray]:
    """Arrays in template order (looked up by name, not by position)."""
    return [record[name].numpy() for name in names]


def local_update(
    cfg: Config,
    fs: FeatureSet,
    global_weights: list[np.ndarray],
    partition_id: int,
    server_round: int,
    seed: int,
    lr: float,
    local_epochs: int,
    proximal_mu: float = 0.0,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """One round of local training. Pure function of the client's data and the global model."""
    train = fs.part("train")
    model = make_model(cfg, fs.X.shape[1])
    set_weights(model, global_weights)
    info = train_epochs(model, train.X, train.y, cfg, local_epochs, seed=derive_seed(seed, "local", server_round, partition_id),
                        lr=lr, proximal_mu=proximal_mu)
    return get_weights(model), {"num-examples": int(len(train)), "train-loss": float(info["loss"]), "steps": int(info["steps"])}


def local_evaluation(cfg: Config, fs: FeatureSet, weights: list[np.ndarray], n_bins: int) -> dict[str, Any]:
    """Score histograms of a model on the client's validation and test rows."""
    model = make_model(cfg, fs.X.shape[1])
    set_weights(model, weights)
    val, test = fs.part("val"), fs.part("test")
    p_val = predict_proba(model, val.X)
    p_test = predict_proba(model, test.X)
    loss = 0.0
    if len(val):
        with torch.no_grad():
            model.eval()
            loss = float(make_loss(cfg)(model(torch.as_tensor(val.X)), torch.as_tensor(val.y.astype(np.float32))))
    return {
        "num-examples": int(len(val)),
        "val-loss": loss,
        "hist-val": ScoreHist.from_scores(p_val, val.typology, n_bins).to_list(),
        "hist-test": ScoreHist.from_scores(p_test, test.typology, n_bins).to_list(),
    }


def make_client_app(cfg: Config, data_dir: str | Path, seed: int, options: dict[str, Any] | None = None) -> ClientApp:
    """Build the ClientApp for one federated run.

    ``data_dir`` is the scenario directory; a node reads ``clients/client_<id>.npz``
    for its own partition id and nothing else. ``options`` carries the optional
    extensions (secure aggregation, DP-SGD, the poisoning demo).
    """
    data_dir = str(data_dir)
    options = dict(options or {})
    n_bins = int(cfg.fl.hist_bins)
    app = ClientApp()

    def _setup(context: Context) -> tuple[int, FeatureSet, list[str]]:
        torch.set_num_threads(1)
        pid = int(context.node_config["partition-id"])
        fs = client_data(data_dir, pid)
        names = [name for name, _, _ in weight_spec(make_model(cfg, fs.X.shape[1]))]
        return pid, fs, names

    @app.query()
    def feature_statistics(msg: Message, context: Context) -> Message:
        pid, fs, _ = _setup(context)
        n, total, total_sq = feature_moments(fs.part("train").X)
        metrics = MetricRecord({"num-examples": int(n), "partition-id": pid, "feat-sum": [float(v) for v in total], "feat-sumsq": [float(v) for v in total_sq]})
        return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)

    @app.train()
    def train(msg: Message, context: Context) -> Message:
        from tbml.fl import extensions

        pid, fs, names = _setup(context)
        config = msg.content["config"]
        server_round = int(config["server-round"])
        global_weights = from_array_record(msg.content["arrays"], names)
        if options.get("dp"):
            weights, metrics = extensions.dp_local_update(cfg, fs, global_weights, pid, server_round, seed, config, options["dp"])
        else:
            weights, metrics = local_update(cfg, fs, global_weights, pid, server_round, seed, lr=float(config["lr"]),
                                            local_epochs=int(config["local-epochs"]), proximal_mu=float(config.get("proximal-mu", 0.0)))
        if options.get("poison") and pid in options["poison"]["clients"]:
            weights = extensions.poison_update(global_weights, weights, options["poison"])
        metrics["partition-id"] = pid
        if options.get("secagg"):
            record = extensions.mask_update(names, weights, metrics, pid, server_round, config, options["secagg"])
            metrics = {k: v for k, v in metrics.items() if k != "num-examples"} | {"num-examples": 1}
        else:
            record = to_array_record(names, weights)
        digest = hash_arrays((name, array.numpy()) for name, array in record.items())
        content = RecordDict({"arrays": record, "metrics": MetricRecord(metrics), "meta": ConfigRecord({"update-sha256": digest})})
        return Message(content=content, reply_to=msg)

    @app.evaluate()
    def evaluate(msg: Message, context: Context) -> Message:
        pid, fs, names = _setup(context)
        metrics = local_evaluation(cfg, fs, from_array_record(msg.content["arrays"], names), n_bins)
        metrics["partition-id"] = pid
        return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)

    return app
