"""Tests for the Phase 0 plumbing.

These assert behaviour, not existence. `assert model is not None` tells you
nothing; `assert the loss went down` tells you the loop works.
"""

from __future__ import annotations

import pytest
import torch

from cirrus.config import Config
from cirrus.device import get_device
from cirrus.models.dummy import DummyModel
from cirrus.train.loop import train


def test_device_is_usable() -> None:
    """Whatever device we detect, a tensor can actually live on it."""
    device = get_device()
    x = torch.ones(2, 2, device=device)
    assert x.sum().item() == pytest.approx(4.0)


def test_unknown_config_key_is_rejected(tmp_path) -> None:
    """A typo in a config must fail loudly, not silently use the default."""
    path = tmp_path / "bad.yaml"
    path.write_text("learning_rat: 0.1\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        Config.from_yaml(path)


def test_model_preserves_shape() -> None:
    """Field in, field out, same grid."""
    model = DummyModel(n_channels=4, hidden_dim=8)
    x = torch.randn(2, 4, 32, 64)
    assert model(x).shape == x.shape


def test_model_can_overfit_one_batch() -> None:
    """The single most useful test in machine learning.

    Given a target the architecture can represent, the loop must drive the
    loss on one fixed batch to near zero. If it cannot, the model, loss or
    optimiser is broken and no amount of data will save you.

    The target is a linear mixing of the input channels — well within the
    capacity of two convolutions. Note what this test deliberately avoids:
    an unrelated random target, which a weight-sharing conv model cannot
    memorise no matter how long you train it.
    """
    torch.manual_seed(0)
    model = DummyModel(n_channels=2, hidden_dim=32)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()

    x = torch.randn(4, 2, 16, 16)
    mixing = torch.tensor([[0.8, -0.4], [0.3, 0.9]])
    y = torch.einsum("bchw,cd->bdhw", x, mixing)

    first = loss_fn(model(x), y).item()
    for _ in range(400):
        optimiser.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimiser.step()

    assert loss.item() < 0.05 * first


def test_training_reduces_loss() -> None:
    """End to end: the tiny config must actually learn something."""
    cfg = Config(n_samples=128, epochs=3, device="cpu")
    history = train(cfg)
    assert history[-1].train_loss < history[0].train_loss


def test_training_is_reproducible() -> None:
    """Same seed, same result. Without this, ablations mean nothing."""
    cfg = Config(n_samples=128, epochs=2, seed=42, device="cpu")
    first = train(cfg)
    second = train(cfg)
    assert first[-1].train_loss == pytest.approx(second[-1].train_loss, rel=1e-6)
