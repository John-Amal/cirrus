"""Prediction heads for precipitation.

Four objectives are compared on the same frozen backbone, which is the
experiment this project exists to run:

- **MSE** and **L1** point heads, the standard practice baselines.
- **CRPS** on a censored shifted gamma distribution.
- **Threshold-weighted CRPS** on the same distribution -- the tail-aware
  objective under test.

All of them predict precipitation in **millimetres**, not in the transformed
space the backbone was pretrained on. Phase 2 measured why: a model unbiased
in ``log1p`` space came out 24% dry in millimetres, because under-dispersion
is amplified by the convex inverse transform. Training the head in physical
units removes that mechanism from the comparison entirely, so any remaining
difference between objectives is attributable to the objective.

The censored shifted gamma (Scheuerer and Hamill, 2015) fits precipitation
well and handles its awkward shape in one object: ``Y = max(0, G + delta)``
with ``G`` gamma-distributed and ``delta`` negative. The censoring puts a
point mass at zero, so there is no need for a separate rain/no-rain
classifier with its own loss and its own calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Gamma

from cirrus.models.attention import TransformerBlock
from cirrus.models.patch_embed import unpatchify

MIN_PARAM = 1e-4  # keeps shape and scale away from a degenerate zero


@dataclass
class Csgd:
    """A censored shifted gamma distribution, per grid cell.

    ``shape`` and ``scale`` parameterise the underlying gamma; ``shift`` is
    negative, and everything the shift pushes below zero becomes dry.
    """

    shape: torch.Tensor
    scale: torch.Tensor
    shift: torch.Tensor

    def sample(
        self, n_samples: int, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Draw ``(..., n_samples)`` values, differentiably.

        ``rsample`` uses implicit reparameterisation, so gradients reach both
        gamma parameters. The clamp is what censors: values pushed below zero
        by the shift become exactly zero, and receive no gradient -- correct,
        since moving a dry point further into the dry region changes nothing.
        """
        expanded = [
            p.unsqueeze(-1).expand(*p.shape, n_samples)
            for p in (self.shape, self.scale, self.shift)
        ]
        gamma = Gamma(concentration=expanded[0], rate=1.0 / expanded[1])
        draws = (
            gamma.rsample() if generator is None else _seeded_rsample(gamma, generator)
        )
        return torch.clamp(draws + expanded[2], min=0.0)

    def probability_of_zero(self, n_samples: int = 200) -> torch.Tensor:
        """Fraction of mass at zero, estimated by sampling.

        The exact value is the gamma CDF at ``-shift``, which PyTorch can
        evaluate but not differentiate with respect to the shape. This is for
        diagnostics, not for training, so sampling is fine.
        """
        with torch.no_grad():
            return (self.sample(n_samples) <= 0).to(self.shape.dtype).mean(dim=-1)

    @property
    def mean_of_gamma(self) -> torch.Tensor:
        """Mean of the underlying (uncensored, unshifted) gamma."""
        return self.shape * self.scale


def _seeded_rsample(gamma: Gamma, generator: torch.Generator) -> torch.Tensor:
    """Reproducible draws for tests.

    ``Gamma.rsample`` takes no generator, so a seeded draw is made by setting
    the global state around the call. Only used in tests and diagnostics.
    """
    state = torch.random.get_rng_state()
    torch.random.manual_seed(
        int(torch.randint(0, 2**31 - 1, (1,), generator=generator))
    )
    try:
        return gamma.rsample()
    finally:
        torch.random.set_rng_state(state)


class GriddedHead(nn.Module):
    """Map backbone tokens to a gridded field of ``out_channels``.

    Optionally runs a few transformer blocks first. With ``depth=0`` this is
    a linear probe, which is the cleanest way to ask what the frozen backbone
    already represents.
    """

    def __init__(
        self,
        dim: int,
        patch: int,
        grid: tuple[int, int],
        out_channels: int = 1,
        depth: int = 2,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.patch = patch
        self.grid = grid
        self.out_channels = out_channels
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, n_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.project = nn.Linear(dim, out_channels * patch * patch)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Tokens ``(B, N, D)`` to a field ``(B, out_channels, H, W)``."""
        for block in self.blocks:
            tokens = block(tokens)
        projected: torch.Tensor = self.project(self.norm(tokens))
        return unpatchify(projected, self.patch, self.grid, self.out_channels)


class PointHead(nn.Module):
    """A single predicted value per grid cell, for the MSE and L1 baselines.

    The output passes through softplus because precipitation cannot be
    negative. Without it the baselines would be free to predict negative
    rain, and would lose to the distributional heads for a reason that has
    nothing to do with the objective being tested.
    """

    def __init__(self, dim: int, patch: int, grid: tuple[int, int], depth: int = 2):
        super().__init__()
        self.head = GriddedHead(dim, patch, grid, out_channels=1, depth=depth)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Predicted precipitation in mm, ``(B, H, W)``."""
        raw: torch.Tensor = self.head(tokens)[:, 0]
        return nn.functional.softplus(raw)


class CsgdHead(nn.Module):
    """Predict a censored shifted gamma per grid cell.

    Three raw channels become shape, scale and shift. Shape and scale are
    softplus-constrained to stay positive; the shift is constrained to be
    negative, which is what creates the dry mass. Leaving the shift free
    would let the model produce a distribution with no mass at zero, which
    cannot represent a dry grid cell at all.
    """

    def __init__(self, dim: int, patch: int, grid: tuple[int, int], depth: int = 2):
        super().__init__()
        self.head = GriddedHead(dim, patch, grid, out_channels=3, depth=depth)

    def forward(self, tokens: torch.Tensor) -> Csgd:
        """Return the per-cell distribution, ``(B, H, W)`` in each parameter."""
        raw = self.head(tokens)
        softplus = nn.functional.softplus
        return Csgd(
            shape=softplus(raw[:, 0]) + MIN_PARAM,
            scale=softplus(raw[:, 1]) + MIN_PARAM,
            shift=-softplus(raw[:, 2]),
        )
