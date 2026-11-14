"""Vertical FL demo: a split neural network for the Bank / Trade registry / Customs picture.

In the vertical setting every party holds different columns of the *same*
trades (aligned beforehand by private set intersection on the invoice number;
the demo uses the shared key directly):

* **bank**     payments and KYC: payer, timing, currency, counterparty registry
* **trade**    invoices and bills of lading: price, quantities, route, trade graph
* **customs**  declarations: declared HS code, value, quantity, weight

Each party runs its own small "bottom" network on its own columns and sends an
embedding per trade. A coordinator that holds the labels runs the "top" network
on the concatenated embeddings and returns the gradient of each embedding. Raw
columns never leave a party.

Indicators that compare two parties' fields directly (invoice against customs,
invoice against payment) cannot be computed by anyone in this setting and are
left out; the top network has to learn such comparisons from the embeddings.
The party blocks therefore include a few raw magnitudes (log value, log
quantity, dates) for it to work with.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from tbml.config import Config
from tbml.data.generator import TradeDataset
from tbml.eval.metrics import evaluate_scores
from tbml.features.build import FeatureSet, Reference
from tbml.features.dictionary import FEATURES
from tbml.features.link import link_dataset
from tbml.models.mlp import feature_moments, fit_mlp, make_loss, moments_to_scaler, predict_proba
from tbml.utils import derive_seed

PARTIES = ["bank", "trade", "customs"]


def _log10(values: Any) -> np.ndarray:
    return np.log10(np.maximum(np.nan_to_num(np.asarray(values, dtype=np.float64)), 0.0) + 1.0)


def party_blocks(ds: TradeDataset, fs: FeatureSet, cfg: Config) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Feature block of each party for the trades in ``fs`` (same row order)."""
    linked = link_dataset(ds).set_index("trade_id").reindex(fs.trade_id)
    ref = Reference.from_tables(ds.tables)
    start = pd.Timestamp(cfg.data.start_date)
    index = {name: j for j, name in enumerate(fs.feature_names)}

    def owned(owner: str) -> tuple[list[np.ndarray], list[str]]:
        names = [f.name for f in FEATURES if f.owner == owner]
        return [fs.X[:, index[n]].astype(np.float64) for n in names], names

    def days(col: str) -> np.ndarray:
        return np.nan_to_num((pd.to_datetime(linked[col]) - start).dt.days.to_numpy(dtype=np.float64)) / 365.0

    blocks: dict[str, tuple[np.ndarray, list[str]]] = {}

    cols, names = owned("bank")
    cols += [_log10(linked["paid_usd"]), days("first_payment_date")]
    names += ["paid_usd_log", "payment_day"]
    blocks["bank"] = (np.column_stack(cols).astype(np.float32), names)

    cols, names = owned("trade")
    hs = ref.hs.reindex(linked["hs_code"])
    cols += [_log10(linked["quantity"]), days("invoice_date"), _log10(linked["shipped_quantity"]), _log10(linked["gross_weight_kg"]),
             (linked["has_shipment"].to_numpy() == 0).astype(np.float64), np.log10(hs["ref_price_usd"].to_numpy(dtype=np.float64))]
    names += ["invoice_qty_log", "invoice_day", "shipped_qty_log", "ship_weight_log", "no_shipment_record", "hs_ref_price_log"]
    blocks["trade"] = (np.column_stack(cols).astype(np.float32), names)

    has = linked["has_customs"].to_numpy() == 1
    dhs = ref.hs.reindex(linked["declared_hs_code"])
    known = has & dhs["ref_price_usd"].notna().to_numpy()
    qty = np.nan_to_num(linked["declared_quantity"].to_numpy(dtype=np.float64))
    value = np.nan_to_num(linked["declared_value_usd"].to_numpy(dtype=np.float64))
    weight = np.nan_to_num(linked["declared_weight_kg"].to_numpy(dtype=np.float64))
    ok = known & (qty > 0) & (value > 0)
    unit_z = np.zeros(len(linked))
    unit_z[ok] = np.clip(np.log(value[ok] / qty[ok] / dhs["ref_price_usd"].to_numpy(dtype=np.float64)[ok]) / dhs["ref_log_sigma"].to_numpy(dtype=np.float64)[ok], -10, 10)
    okw = known & (qty > 0) & (weight > 0)
    weight_dev = np.zeros(len(linked))
    weight_dev[okw] = np.clip(np.log(weight[okw] / qty[okw] / dhs["kg_per_unit"].to_numpy(dtype=np.float64)[okw]), -6, 6)
    cols = [has.astype(np.float64), _log10(value), _log10(qty), _log10(weight), unit_z, np.abs(unit_z), weight_dev, np.abs(weight_dev),
            (has & ~known).astype(np.float64), np.where(known, np.log10(dhs["ref_price_usd"].fillna(1.0).to_numpy(dtype=np.float64)), 0.0), days("declaration_date")]
    names = ["has_declaration", "declared_value_log", "declared_qty_log", "declared_weight_log", "declared_unit_value_z", "declared_unit_value_abs_z",
             "declared_weight_per_unit_dev", "declared_weight_per_unit_abs_dev", "declared_hs_not_in_reference", "declared_hs_ref_price_log", "declaration_day"]
    blocks["customs"] = (np.column_stack(cols).astype(np.float32), names)
    return blocks


class Bottom(nn.Module):
    """A party's own network: its columns in, an embedding out."""

    def __init__(self, n_features: int, hidden: int, embedding: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, embedding), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.clamp((x - self.mean) / self.std, -8.0, 8.0))


class Top(nn.Module):
    """The coordinator's network on the concatenated embeddings."""

    def __init__(self, n_inputs: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_inputs, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class SplitNN:
    """Split neural network with explicit message passing between parties and coordinator."""

    def __init__(self, cfg: Config, sizes: dict[str, int], seed: int) -> None:
        v = cfg.fl.vertical
        torch.manual_seed(int(seed) % (2**31))
        self.parties = list(sizes)
        self.embedding = int(v.embedding)
        self.bottoms = {p: Bottom(sizes[p], int(v.hidden), self.embedding) for p in self.parties}
        self.top = Top(self.embedding * len(self.parties), int(v.hidden))
        self.loss_fn = make_loss(cfg)
        self.batch = int(cfg.models.mlp.batch_size)
        lr = float(v.lr)
        self.optimizers = {p: torch.optim.SGD(self.bottoms[p].parameters(), lr=lr, momentum=0.9) for p in self.parties}
        self.top_optimizer = torch.optim.SGD(self.top.parameters(), lr=lr, momentum=0.9)
        self.bytes_up = 0  # embeddings, parties -> coordinator
        self.bytes_down = 0  # gradients of the embeddings, coordinator -> parties

    def fit_scalers(self, blocks: dict[str, np.ndarray]) -> None:
        """Each party standardises its own columns from its own training rows."""
        for p in self.parties:
            mean, std = moments_to_scaler(*feature_moments(blocks[p]))
            self.bottoms[p].mean.copy_(torch.as_tensor(mean))
            self.bottoms[p].std.copy_(torch.as_tensor(std))

    def train_epoch(self, blocks: dict[str, torch.Tensor], y: torch.Tensor, generator: torch.Generator) -> float:
        for module in [*self.bottoms.values(), self.top]:
            module.train()
        order = torch.randperm(len(y), generator=generator)
        total, steps = 0.0, 0
        for start in range(0, len(order), self.batch):
            idx = order[start : start + self.batch]
            if len(idx) < 2:
                continue
            # 1. every party computes embeddings of the batch on its own columns
            local = {p: self.bottoms[p](blocks[p][idx]) for p in self.parties}
            # 2. only the embedding values cross the boundary to the coordinator
            received = {p: local[p].detach().requires_grad_(True) for p in self.parties}
            self.bytes_up += sum(e.numel() * 4 for e in received.values())
            # 3. the coordinator (label holder) computes the loss and updates the top network
            self.top_optimizer.zero_grad()
            loss = self.loss_fn(self.top(torch.cat([received[p] for p in self.parties], dim=1)), y[idx])
            loss.backward()
            self.top_optimizer.step()
            # 4. it returns the gradient with respect to each embedding; parties finish back-propagation locally
            for p in self.parties:
                self.bytes_down += received[p].grad.numel() * 4
                self.optimizers[p].zero_grad()
                local[p].backward(received[p].grad)
                self.optimizers[p].step()
            total += float(loss.detach())
            steps += 1
        return total / max(steps, 1)

    @torch.no_grad()
    def predict(self, blocks: dict[str, np.ndarray]) -> np.ndarray:
        for module in [*self.bottoms.values(), self.top]:
            module.eval()
        z = torch.cat([self.bottoms[p](torch.as_tensor(blocks[p])) for p in self.parties], dim=1)
        return torch.sigmoid(self.top(z)).numpy().astype(np.float64)

    def state(self) -> dict[str, Any]:
        return {"top": {k: v.clone() for k, v in self.top.state_dict().items()},
                "bottoms": {p: {k: v.clone() for k, v in b.state_dict().items()} for p, b in self.bottoms.items()}}

    def load(self, state: dict[str, Any]) -> None:
        self.top.load_state_dict(state["top"])
        for p, b in self.bottoms.items():
            b.load_state_dict(state["bottoms"][p])


def run_vertical_demo(cfg: Config, seed: int, ds: TradeDataset, fs: FeatureSet) -> dict[str, Any]:
    """Split NN against each party alone and against a centralised model on the same columns."""
    blocks = party_blocks(ds, fs, cfg)
    masks = {name: fs.split == name for name in ("train", "val", "test")}
    part = {name: {p: blocks[p][0][m] for p in PARTIES} for name, m in masks.items()}
    y = {name: fs.y[m] for name, m in masks.items()}
    typ = {name: fs.typology[m] for name, m in masks.items()}
    metric = str(cfg.models.threshold_metric)
    epochs = int(cfg.fl.vertical.epochs)

    model = SplitNN(cfg, {p: blocks[p][0].shape[1] for p in PARTIES}, derive_seed(seed, "vertical"))
    with torch.no_grad():
        model.fit_scalers(part["train"])
    tensors = {p: torch.as_tensor(part["train"][p]) for p in PARTIES}
    labels = torch.as_tensor(y["train"].astype(np.float32))
    generator = torch.Generator().manual_seed(derive_seed(seed, "vertical-batches") % (2**31))
    best_score, best_state, best_epoch, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        loss = model.train_epoch(tensors, labels, generator)
        score = float(average_precision_score(y["val"], model.predict(part["val"])))
        history.append({"epoch": epoch, "loss": loss, "val_pr_auc": score, "bytes_cumulative": model.bytes_up + model.bytes_down})
        if score > best_score:
            best_score, best_state, best_epoch = score, model.state(), epoch
    model.load(best_state)
    results: dict[str, Any] = {"split_nn": evaluate_scores(model.predict(part["val"]), typ["val"], model.predict(part["test"]), typ["test"], metric)}
    results["split_nn"]["best_epoch"] = best_epoch

    # each party on its own (it would need the labels, which is generous to this baseline)
    for p in PARTIES:
        alone, _ = fit_mlp(cfg, part["train"][p], y["train"], part["val"][p], y["val"], seed=derive_seed(seed, "vertical-alone", p) % (2**31), epochs=epochs)
        results[f"{p}_alone"] = evaluate_scores(predict_proba(alone, part["val"][p]), typ["val"], predict_proba(alone, part["test"][p]), typ["test"], metric)
    # one model on all party columns pooled: what the split network could reach at best
    pooled = {name: np.concatenate([part[name][p] for p in PARTIES], axis=1) for name in part}
    central, _ = fit_mlp(cfg, pooled["train"], y["train"], pooled["val"], y["val"], seed=derive_seed(seed, "vertical-pooled") % (2**31), epochs=epochs)
    results["pooled_columns"] = evaluate_scores(predict_proba(central, pooled["val"]), typ["val"], predict_proba(central, pooled["test"]), typ["test"], metric)

    n_train = int(masks["train"].sum())
    per_epoch = n_train * model.embedding * 4 * 2 * len(PARTIES)
    return {
        "seed": seed,
        "parties": {p: {"n_features": int(blocks[p][0].shape[1]), "features": blocks[p][1]} for p in PARTIES},
        "epochs": epochs,
        "embedding": model.embedding,
        "methods": results,
        "history": history,
        "communication": {"bytes_up": model.bytes_up, "bytes_down": model.bytes_down, "bytes_total": model.bytes_up + model.bytes_down,
                          "bytes_per_epoch": per_epoch, "bytes_per_trade_per_epoch": model.embedding * 4 * 2 * len(PARTIES),
                          "bytes_until_best_epoch": per_epoch * best_epoch, "n_train_trades": n_train},
    }
