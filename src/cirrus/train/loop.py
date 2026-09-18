"""The training loop.

Written by hand rather than delegated to a framework. Everything that goes
wrong in ML training goes wrong here, and you cannot debug what you have not
read.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cirrus.config import Config
from cirrus.device import device_report, get_device
from cirrus.models.dummy import DummyModel


@dataclass
class EpochResult:
    """What one epoch produced."""

    epoch: int
    train_loss: float
    val_loss: float
    seconds: float


def set_seed(seed: int) -> None:
    """Seed every source of randomness we use.

    Note this does not make CUDA fully deterministic; it makes runs
    *comparable*, which is what we need for ablations.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_synthetic_loaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    """Random tensors shaped like ERA5 fields.

    The target is a fixed linear mixing of the input channels plus noise, so
    the task is learnable and the loss *must* fall. A loop that cannot fit
    this is broken, and you want to discover that now rather than after a
    six-hour pretraining run.

    Replaced in Phase 1 by the real ERA5 dataset.
    """
    shape = (cfg.n_samples, cfg.n_channels, cfg.grid_height, cfg.grid_width)
    x = torch.randn(*shape)

    mixing = torch.randn(cfg.n_channels, cfg.n_channels) * 0.5
    y = torch.einsum("bchw,cd->bdhw", x, mixing) + 0.1 * torch.randn(*shape)

    n_train = int(0.8 * cfg.n_samples)
    train = TensorDataset(x[:n_train], y[:n_train])
    val = TensorDataset(x[n_train:], y[n_train:])

    return (
        DataLoader(train, batch_size=cfg.batch_size, shuffle=True),
        DataLoader(val, batch_size=cfg.batch_size, shuffle=False),
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Mean loss over a loader, with gradients off."""
    model.eval()
    total, n_batches = 0.0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            total += loss_fn(model(inputs), targets).item()
            n_batches += 1
    return total / max(n_batches, 1)


def train(cfg: Config, checkpoint_dir: Path | None = None) -> list[EpochResult]:
    """Run training end to end and return the per-epoch history."""
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"device: {device_report(device)}")

    train_loader, val_loader = make_synthetic_loaders(cfg)
    model = DummyModel(cfg.n_channels, cfg.hidden_dim).to(device)
    print(f"model: {model.n_parameters:,} parameters")

    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()

    history: list[EpochResult] = []
    for epoch in range(1, cfg.epochs + 1):
        started = time.perf_counter()
        model.train()
        running, n_batches = 0.0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # The four lines that are the whole of gradient descent:
            optimiser.zero_grad()  # clear gradients from the previous step
            loss = loss_fn(model(inputs), targets)  # forward pass
            loss.backward()  # backward pass: fill .grad on every parameter
            optimiser.step()  # move each parameter downhill

            running += loss.item()
            n_batches += 1

        result = EpochResult(
            epoch=epoch,
            train_loss=running / max(n_batches, 1),
            val_loss=evaluate(model, val_loader, loss_fn, device),
            seconds=time.perf_counter() - started,
        )
        history.append(result)
        print(
            f"epoch {result.epoch:>3}  "
            f"train {result.train_loss:.4f}  "
            f"val {result.val_loss:.4f}  "
            f"{result.seconds:.1f}s"
        )

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"{cfg.name}.pt"
        torch.save(
            {"config": cfg.to_dict(), "model_state": model.state_dict()},
            path,
        )
        print(f"saved checkpoint: {path}")

    return history
