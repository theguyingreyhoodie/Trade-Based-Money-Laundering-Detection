"""The neural model: a small MLP on the risk indicators.

The same network is trained centrally (baseline), by each client alone
(local-only) and with federated averaging. Feature standardisation is part of
the model (non-trainable buffers), so one set of arrays fully describes a model
and FedAvg can move it around without a separate scaler.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from tbml.config import Config

STD_CLIP = 8.0


class Standardize(nn.Module):
    """(x - mean) / std with clipping; mean and std are buffers, not parameters."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / self.std, -STD_CLIP, STD_CLIP)


class TBMLNet(nn.Module):
    """Standardise -> hidden layers (ReLU, dropout) -> one logit."""

    def __init__(self, n_features: int, hidden: Sequence[int] = (64, 32), dropout: float = 0.1) -> None:
        super().__init__()
        self.scaler = Standardize(n_features)
        layers: list[nn.Module] = []
        width = n_features
        for h in hidden:
            layers += [nn.Linear(width, int(h)), nn.ReLU(), nn.Dropout(dropout)]
            width = int(h)
        layers.append(nn.Linear(width, 1))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(self.scaler(x)).squeeze(-1)


def make_model(cfg: Config, n_features: int, seed: int | None = None) -> TBMLNet:
    """Fresh model; ``seed`` fixes the weight initialisation."""
    if seed is not None:
        torch.manual_seed(int(seed))
    m = cfg.models.mlp
    return TBMLNet(n_features, hidden=tuple(m.hidden), dropout=float(m.dropout))


def feature_moments(X: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """(count, sum, sum of squares) per feature: the aggregates a client shares for standardisation."""
    X64 = np.asarray(X, dtype=np.float64)
    return len(X64), X64.sum(axis=0), (X64**2).sum(axis=0)


def moments_to_scaler(n: float, total: np.ndarray, total_sq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation from (possibly aggregated) moments."""
    mean = np.asarray(total, dtype=np.float64) / max(n, 1.0)
    var = np.maximum(np.asarray(total_sq, dtype=np.float64) / max(n, 1.0) - mean**2, 0.0)
    std = np.sqrt(var)
    std[std < 1e-6] = 1.0  # constant feature: leave it centred, do not blow it up
    return mean.astype(np.float32), std.astype(np.float32)


def set_scaler(model: TBMLNet, mean: np.ndarray, std: np.ndarray) -> None:
    model.scaler.mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
    model.scaler.std.copy_(torch.as_tensor(std, dtype=torch.float32))


def get_weights(model: nn.Module) -> list[np.ndarray]:
    """All arrays of the model (parameters and buffers) in ``state_dict`` order."""
    return [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]


def set_weights(model: nn.Module, weights: Sequence[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    if len(keys) != len(weights):
        raise ValueError(f"expected {len(keys)} arrays, got {len(weights)}")
    state = {k: torch.as_tensor(np.asarray(w)).to(model.state_dict()[k].dtype).reshape(model.state_dict()[k].shape) for k, w in zip(keys, weights)}
    model.load_state_dict(state, strict=True)


def weight_spec(model: nn.Module) -> list[tuple[str, tuple[int, ...], str]]:
    """(name, shape, dtype) of every array: the template the FL server checks updates against."""
    return [(k, tuple(v.shape), str(v.detach().cpu().numpy().dtype)) for k, v in model.state_dict().items()]


def trainable_mask(model: nn.Module) -> list[bool]:
    """True for arrays that are trained (parameters), False for buffers such as the scaler."""
    params = {name for name, _ in model.named_parameters()}
    return [k in params for k in model.state_dict().keys()]


def make_loss(cfg: Config):
    """Class-imbalance aware loss: weighted BCE (default) or focal loss."""
    m = cfg.models.mlp
    if m.loss == "focal":
        gamma, alpha = float(m.focal_gamma), float(m.focal_alpha)

        def focal(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            bce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
            p_t = torch.exp(-bce)
            weight = torch.where(y > 0.5, alpha, 1.0 - alpha)
            return (weight * (1.0 - p_t) ** gamma * bce).mean()

        return focal
    pos_weight = torch.tensor(float(m.pos_weight))

    def weighted_bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)

    return weighted_bce


def make_optimizer(cfg: Config, model: nn.Module, lr: float | None = None) -> torch.optim.Optimizer:
    m = cfg.models.mlp
    lr = float(m.lr if lr is None else lr)
    if m.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=float(m.weight_decay))
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=float(m.weight_decay))


def train_epochs(
    model: TBMLNet,
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config,
    epochs: int,
    seed: int,
    lr: float | None = None,
    proximal_mu: float = 0.0,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Run ``epochs`` passes of mini-batch training in place.

    ``proximal_mu > 0`` adds the FedProx term ``mu/2 * ||w - w_start||^2`` that
    keeps a client's model close to the global model it started the round from.
    """
    if len(X) == 0 or epochs <= 0:
        return {"loss": 0.0, "steps": 0}
    loss_fn = make_loss(cfg)
    opt = optimizer or make_optimizer(cfg, model, lr)
    xb = torch.as_tensor(np.asarray(X, dtype=np.float32))
    yb = torch.as_tensor(np.asarray(y, dtype=np.float32))
    batch = int(cfg.models.mlp.batch_size)
    torch.manual_seed(int(seed) % (2**31))  # dropout masks come from the global generator
    gen = torch.Generator().manual_seed(int(seed) % (2**31))
    anchor = [p.detach().clone() for p in model.parameters()] if proximal_mu > 0 else None
    model.train()
    total, steps = 0.0, 0
    for _ in range(int(epochs)):
        order = torch.randperm(len(xb), generator=gen)
        for start in range(0, len(order), batch):
            idx = order[start : start + batch]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = loss_fn(model(xb[idx]), yb[idx])
            if anchor is not None:
                loss = loss + 0.5 * proximal_mu * sum(((p - a) ** 2).sum() for p, a in zip(model.parameters(), anchor))
            loss.backward()
            opt.step()
            total += float(loss.detach())
            steps += 1
    return {"loss": total / max(steps, 1), "steps": steps}


@torch.no_grad()
def predict_proba(model: TBMLNet, X: np.ndarray, batch: int = 8192) -> np.ndarray:
    """P(illicit) for each row."""
    model.eval()
    out = []
    for start in range(0, len(X), batch):
        xb = torch.as_tensor(np.asarray(X[start : start + batch], dtype=np.float32))
        out.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def fit_mlp(
    cfg: Config,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    epochs: int | None = None,
    scaler: tuple[np.ndarray, np.ndarray] | None = None,
    init_weights: Sequence[np.ndarray] | None = None,
    lr: float | None = None,
) -> tuple[TBMLNet, dict[str, Any]]:
    """Train on one data set with early stopping on validation PR-AUC.

    Used for the centralised baseline and for local-only client models. The
    scaler is fitted on the training rows only unless one is passed in.

    ``init_weights`` warm-starts from an existing model (for example the global
    FL model, including its scaler); the starting model then counts as "epoch 0"
    and is kept if no later epoch validates better.
    """
    from sklearn.metrics import average_precision_score

    m = cfg.models.mlp
    epochs = int(m.epochs if epochs is None else epochs)
    model = make_model(cfg, X_train.shape[1], seed)
    if init_weights is not None:
        set_weights(model, init_weights)
    else:
        mean, std = scaler if scaler is not None else moments_to_scaler(*feature_moments(X_train))
        with torch.no_grad():
            set_scaler(model, mean, std)
    opt = make_optimizer(cfg, model, lr)
    can_validate = len(y_val) > 0 and 0 < y_val.sum() < len(y_val)
    best_score, best_state, best_epoch, stale = -1.0, None, 0, 0
    history: list[dict[str, float]] = []
    if init_weights is not None and can_validate:
        best_score = float(average_precision_score(y_val, predict_proba(model, X_val)))
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for epoch in range(1, epochs + 1):
        info = train_epochs(model, X_train, y_train, cfg, 1, seed=seed * 1000 + epoch, optimizer=opt)
        score = float(average_precision_score(y_val, predict_proba(model, X_val))) if can_validate else -float(info["loss"])
        history.append({"epoch": epoch, "loss": info["loss"], "val_score": score})
        if score > best_score or best_state is None:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(m.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "epochs_run": len(history), "best_val_score": best_score, "history": history}
