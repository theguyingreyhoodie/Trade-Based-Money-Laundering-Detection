"""Model: private bank encoder + shared head.

    bank features (input_dim differs per bank)
        -> Encoder (PRIVATE, never leaves the bank)       -> 32-dim representation
        -> SharedHead (identical shape at every bank)     -> laundering logit
                     ^ only these parameters are exchanged with the FL server and FedAvg'd

Used by the Flower client (src/aml_fl/fl/) via get_shared_weights / set_shared_weights.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

LATENT_DIM = 32


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, latent: int = LATENT_DIM, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, latent), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedHead(nn.Module):
    def __init__(self, latent: int = LATENT_DIM, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)  # logits; apply sigmoid for probabilities


class BankModel(nn.Module):
    def __init__(self, input_dim: int, latent: int = LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(input_dim, latent=latent)
        self.head = SharedHead(latent=latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    # --- helpers for federated learning: exchange ONLY the shared head ------------
    def get_shared_weights(self) -> list[np.ndarray]:
        return [p.detach().cpu().numpy().copy() for p in self.head.state_dict().values()]

    def set_shared_weights(self, weights: list[np.ndarray]) -> None:
        keys = list(self.head.state_dict().keys())
        if len(keys) != len(weights):
            raise ValueError(f"Expected {len(keys)} arrays for the shared head, got {len(weights)}")
        state = {k: torch.as_tensor(w) for k, w in zip(keys, weights)}
        self.head.load_state_dict(state, strict=True)


def pos_weight(y: np.ndarray) -> torch.Tensor:
    """Class weight for BCEWithLogitsLoss: (#negatives / #positives)."""
    pos = max(1.0, float(y.sum()))
    return torch.tensor((len(y) - pos) / pos, dtype=torch.float32)


def train_epochs(model: nn.Module, X: np.ndarray, y: np.ndarray, epochs: int = 1, lr: float = 1e-3,
                 batch_size: int = 512, weight: torch.Tensor | None = None,
                 device: str = "cpu") -> float:
    """Plain local training loop. Returns the mean loss of the last epoch."""
    model.to(device).train()
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight.to(device) if weight is not None else None)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    last = 0.0
    for _ in range(epochs):
        total, n = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            n += len(xb)
        last = total / max(n, 1)
    return last


def train_early_stopping(model: nn.Module, X: np.ndarray, y: np.ndarray, X_val: np.ndarray,
                         y_val: np.ndarray, max_epochs: int = 60, patience: int = 8, lr: float = 1e-3,
                         batch_size: int = 512, weight: torch.Tensor | None = None) -> tuple[float, int]:
    """Train up to max_epochs, keep the weights with the best validation PR-AUC.

    Stops after `patience` epochs without improvement. Returns (best val PR-AUC, best epoch).
    """
    from sklearn.metrics import average_precision_score

    model.train()
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_ap, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        ap = average_precision_score(y_val, predict_proba(model, X_val)) if y_val.sum() else 0.0
        if ap > best_ap:
            best_ap, best_epoch = ap, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return float(best_ap), best_epoch


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, batch_size: int = 4096, device: str = "cpu") -> np.ndarray:
    model.to(device).eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
