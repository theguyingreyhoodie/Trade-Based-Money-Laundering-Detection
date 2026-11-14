"""DP-SGD local training with Opacus (stretch goal: privacy / utility trade-off).

Each client trains with per-example gradient clipping and Gaussian noise, so
that its model update is (epsilon, delta)-differentially private with respect to
its own training rows. This protects a single trade record against whoever sees
the update (the server and, through the global model, the other clients).

Accounting. A client draws Poisson batches with rate ``q = batch / n`` and takes
about ``n / batch`` steps per local epoch. Over a whole run it takes
``rounds * local_epochs * n / batch`` steps with the same noise multiplier, and
Renyi-DP composition over all of them gives the client's epsilon. The noise
multiplier that hits a target epsilon is found with Opacus' accountant; it
depends on the client's own ``n``, so every client computes its own.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch

from tbml.config import Config
from tbml.models.mlp import get_weights, make_loss, make_model, make_optimizer, set_weights
from tbml.utils import derive_seed


def total_steps(n: int, batch: int, rounds: int, local_epochs: int) -> int:
    return int(rounds * local_epochs * max(1, int(np.ceil(n / batch))))


def noise_for_epsilon(epsilon: float, delta: float, n: int, batch: int, rounds: int, local_epochs: int) -> float:
    """Noise multiplier with which a client of ``n`` rows spends ``epsilon`` over the whole run."""
    from opacus.accountants.utils import get_noise_multiplier

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(get_noise_multiplier(target_epsilon=float(epsilon), target_delta=float(delta), sample_rate=min(1.0, batch / max(n, 1)),
                                          steps=total_steps(n, batch, rounds, local_epochs), accountant="rdp"))


def spent_epsilon(noise_multiplier: float, delta: float, n: int, batch: int, rounds: int, local_epochs: int) -> float:
    """Epsilon spent by a client after the whole run (Renyi-DP accountant)."""
    from opacus.accountants import RDPAccountant

    accountant = RDPAccountant()
    accountant.history = [(float(noise_multiplier), min(1.0, batch / max(n, 1)), total_steps(n, batch, rounds, local_epochs))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(accountant.get_epsilon(float(delta)))


def dp_train_round(cfg: Config, fs: Any, global_weights: list[np.ndarray], partition_id: int, server_round: int, seed: int, lr: float,
                   local_epochs: int, options: dict[str, Any]) -> tuple[list[np.ndarray], dict[str, Any]]:
    """One round of local DP-SGD starting from the global model."""
    from opacus import PrivacyEngine

    train = fs.part("train")
    n = len(train)
    batch = int(options.get("batch_size") or cfg.models.mlp.batch_size)
    rounds = int(options["rounds"])
    delta = float(options["delta"])
    sigma = noise_for_epsilon(float(options["epsilon"]), delta, n, batch, rounds, local_epochs)
    metrics = {"num-examples": int(n), "train-loss": 0.0, "steps": 0, "dp-noise": float(sigma),
               "dp-epsilon": spent_epsilon(sigma, delta, n, batch, rounds, local_epochs)}
    model = make_model(cfg, fs.X.shape[1])
    set_weights(model, global_weights)
    if n < 2:
        return get_weights(model), metrics

    torch.manual_seed(derive_seed(seed, "dp", server_round, partition_id) % (2**31))
    dataset = torch.utils.data.TensorDataset(torch.as_tensor(train.X), torch.as_tensor(train.y.astype(np.float32)))
    loader = torch.utils.data.DataLoader(dataset, batch_size=min(batch, n))
    optimizer = make_optimizer(cfg, model, lr)
    loss_fn = make_loss(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        private_model, private_optimizer, private_loader = PrivacyEngine(accountant="rdp").make_private(
            module=model, optimizer=optimizer, data_loader=loader, noise_multiplier=sigma, max_grad_norm=float(options["max_grad_norm"]))
        private_model.train()
        total, steps = 0.0, 0
        for _ in range(int(local_epochs)):
            for xb, yb in private_loader:
                if len(xb) == 0:  # an empty Poisson batch
                    continue
                private_optimizer.zero_grad()
                loss = loss_fn(private_model(xb), yb)
                loss.backward()
                private_optimizer.step()
                total += float(loss.detach())
                steps += 1
    metrics.update({"train-loss": total / max(steps, 1), "steps": steps})
    return get_weights(model), metrics  # Opacus trains the wrapped module in place
