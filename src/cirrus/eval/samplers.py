"""One interface for everything that can be scored.

A trained head, persistence and climatology are very different objects, but
every metric in this project needs the same thing from them: samples of
predicted precipitation, ``(batch, lat, lon, draws)``, in millimetres. A
deterministic predictor returns its value repeated.

Expressing them all as the same callable means the scoring code has no
special cases, and -- more importantly -- the baselines go through exactly
the same metric code as the trained arms. A baseline scored by a separate
path is a baseline you cannot fully trust.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from cirrus.eval.baselines import ClimatologyBaseline
from cirrus.models.patch_embed import flatten_time
from cirrus.models.vit import ViT
from cirrus.train.finetune import FinetuneSpec, PrecipitationTarget

# batch -> (batch, lat, lon, draws) of precipitation in mm
Sampler = Callable[[dict[str, torch.Tensor]], torch.Tensor]


def trained_sampler(
    spec: FinetuneSpec,
    head: torch.nn.Module,
    backbone: ViT,
    device: torch.device,
    draws: int | None = None,
) -> Sampler:
    """Build a sampler for a fine-tuned head on the frozen backbone."""
    n_draws = draws if draws is not None else spec.n_samples

    def sample(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = flatten_time(batch["input"]).to(device)
        tokens = backbone(x)
        if spec.is_distributional:
            drawn: torch.Tensor = head(tokens).sample(n_draws)
            return drawn
        point: torch.Tensor = head(tokens)
        return point.unsqueeze(-1).expand(-1, -1, -1, n_draws)

    return sample


def persistence_sampler(
    to_mm: PrecipitationTarget,
    channel: int,
    device: torch.device,
    draws: int = 1,
) -> Sampler:
    """Build a sampler that carries the last observed field forward.

    Its implied climatology is the observed one shifted in time, so its
    return levels are right by construction while its forecast skill is poor.
    That combination is the point: it shows that reproducing the observed
    tail is necessary but nowhere near sufficient.
    """

    def sample(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        last_step = batch["input"][:, -1, channel].to(device)
        return to_mm(last_step).unsqueeze(-1).expand(-1, -1, -1, draws)

    return sample


def climatology_sampler(
    baseline: ClimatologyBaseline,
    grid: tuple[int, int],
    device: torch.device,
    draws: int = 24,
) -> Sampler:
    """Build a sampler drawing from this cell's historical month.

    Ignores the input entirely. A model that cannot beat this has learned
    nothing about the atmosphere, only about geography and season.
    """

    def sample(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return baseline.samples(batch["time"].to(device), grid, draws)

    return sample
