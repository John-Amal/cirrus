"""Masked-autoencoder pretraining.

The objective: hide most of the globe and make the model reconstruct it. To
fill in a masked patch the model has to learn what the atmosphere looks
like -- that pressure and wind are related, that moisture organises into
bands, that a trough here implies a ridge there. No labels are needed, which
is what makes it *self-supervised* and what lets it use every ERA5 timestep.

Three things differ from the standard image MAE, all for domain reasons.

**Only the dynamic channels are reconstructed.** The input also carries
static fields and time encodings. Those are constant or known in advance, so
predicting them is free marks that flatter the loss without teaching
anything.

**The loss is latitude-weighted.** On a 5.625 deg grid a patch at 80 degrees
covers roughly a sixth of the area of one at the equator. Unweighted, the
model would spend a disproportionate share of its capacity on the poles.

**Targets are not per-patch normalised.** The image-MAE trick of normalising
each patch before the loss is unnecessary here: the data was already z-scored
per channel in Phase 1, using latitude-weighted statistics.

The decoder is deliberately small and thrown away after pretraining. Its job
is only to make the encoder's representation good enough to reconstruct from;
it is not part of the model you fine-tune.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from cirrus.config import read_yaml_mapping
from cirrus.models.attention import TransformerBlock
from cirrus.models.patch_embed import patchify, unpatchify
from cirrus.models.vit import ViT

EPS = 1e-8


@dataclass(frozen=True)
class MaeSpec:
    """Pretraining objective settings."""

    mask_ratio: float = 0.75
    decoder_dim: int = 128
    decoder_depth: int = 2
    decoder_heads: int = 4
    latitude_weighted: bool = True

    def __post_init__(self) -> None:
        """Reject ratios that would mask everything or nothing."""
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {self.mask_ratio}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> MaeSpec:
        """Load a spec, rejecting unknown keys."""
        raw: dict[str, Any] = read_yaml_mapping(path, (f.name for f in fields(cls)))
        return cls(**raw)


def gather_tokens(tokens: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Select or reorder tokens per sample.

    ``tokens`` is (B, N, D) and ``index`` is (B, M); the result is (B, M, D)
    where row b takes the tokens named by ``index[b]``. Each sample has its
    own mask, so a single shared index would be wrong.
    """
    width = tokens.shape[-1]
    return torch.gather(tokens, 1, index.unsqueeze(-1).expand(-1, -1, width))


def latitude_token_weights(
    latitudes: torch.Tensor, patch: int, n_patches_lon: int
) -> torch.Tensor:
    """Per-token area weights from cos(latitude), normalised to mean one.

    Normalising to mean one keeps the loss on the same scale as an unweighted
    one, so learning rates transfer between the two.
    """
    per_row = torch.cos(torch.deg2rad(latitudes.to(torch.float32)))
    per_patch_row = per_row.reshape(-1, patch).mean(dim=1)  # (rows,)
    weights = per_patch_row[:, None].expand(-1, n_patches_lon).reshape(-1)
    return weights / weights.mean().clamp(min=EPS)


def masked_patch_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    token_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared error over masked patches only.

    Args:
        prediction: (B, N, P) predicted patch contents.
        target: (B, N, P) true patch contents.
        mask: (B, N), 1 where the patch was hidden from the encoder.
        token_weights: (N,) per-position weights, or None for uniform.

    Scoring visible patches too would let the model earn most of its reward by
    copying what it can already see, which is not the skill being taught.

    """
    per_token = ((prediction - target) ** 2).mean(dim=-1)  # (B, N)
    weights = mask if token_weights is None else mask * token_weights
    total: torch.Tensor = (per_token * weights).sum() / weights.sum().clamp(min=EPS)
    return total


class MaskedAutoencoder(nn.Module):
    """Wraps a backbone with a masking scheme and a lightweight decoder.

    Args:
        backbone: The encoder being pretrained.
        target_indices: Which input channels to reconstruct -- the dynamic
            ones. Statics and time encodings are excluded deliberately.
        spec: Objective settings.
        latitudes: Latitude of each grid row, in degrees. Required when
            ``spec.latitude_weighted`` is set; there is no safe default,
            since guessing the grid would silently mis-weight the loss.

    """

    def __init__(
        self,
        backbone: ViT,
        target_indices: Sequence[int],
        spec: MaeSpec | None = None,
        latitudes: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.spec = spec or MaeSpec()
        self.patch = backbone.spec.patch
        self.grid = backbone.grid
        self.n_tokens = backbone.n_tokens

        n_in = backbone.patch_embed.in_channels
        if not target_indices:
            raise ValueError("target_indices must name at least one channel")
        if max(target_indices) >= n_in or min(target_indices) < 0:
            raise ValueError(
                f"target_indices out of range for {n_in} input channels: "
                f"{sorted(set(target_indices))}"
            )
        self.register_buffer(
            "target_indices", torch.as_tensor(list(target_indices), dtype=torch.long)
        )
        self.n_target_channels = len(target_indices)
        patch_values = self.n_target_channels * self.patch * self.patch

        if self.spec.latitude_weighted:
            if latitudes is None:
                raise ValueError(
                    "latitude_weighted is set but no latitudes were given; "
                    "pass the grid's latitudes or turn the weighting off"
                )
            if latitudes.numel() != self.grid[0]:
                raise ValueError(
                    f"expected {self.grid[0]} latitudes, got {latitudes.numel()}"
                )
            weights = latitude_token_weights(
                latitudes, self.patch, backbone.patch_embed.n_patches_lon
            )
        else:
            weights = torch.ones(self.n_tokens)
        self.register_buffer("token_weights", weights)

        decoder_dim = self.spec.decoder_dim
        self.decoder_embed = nn.Linear(backbone.spec.dim, decoder_dim)
        # One learned vector standing in for "a patch was here". The decoder
        # recovers what belongs there from position and from context.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.n_tokens, decoder_dim)
        )
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(decoder_dim, self.spec.decoder_heads)
                for _ in range(self.spec.decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_values)

    @property
    def n_visible(self) -> int:
        """Tokens the encoder actually sees."""
        return self.n_tokens - self.n_masked

    @property
    def n_masked(self) -> int:
        """Tokens hidden from the encoder."""
        return int(round(self.n_tokens * self.spec.mask_ratio))

    def random_masking(
        self, tokens: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep a random subset of tokens, per sample.

        Implemented by sorting uniform noise: ``argsort`` of random values is
        a random permutation, and taking its first ``n_visible`` entries is a
        uniform random subset without replacement. Sorting that permutation
        again gives ``ids_restore``, the inverse permutation, which the
        decoder uses to put tokens back where they came from.

        Returns:
            ``(visible_tokens, mask, ids_restore)`` where ``mask`` is 1 for
            hidden positions, in the original token order.

        """
        batch, n_tokens, _ = tokens.shape
        noise = torch.rand(batch, n_tokens, device=tokens.device, generator=generator)

        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, : self.n_visible]
        visible = gather_tokens(tokens, ids_keep)

        # Build the mask in shuffled order, where the kept tokens are exactly
        # the first n_visible, then unshuffle it into original order.
        mask = torch.ones(batch, n_tokens, device=tokens.device)
        mask[:, : self.n_visible] = 0.0
        mask = torch.gather(mask, 1, ids_restore)
        return visible, mask, ids_restore

    def decode(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """Fill masked positions and predict patch contents for every token."""
        batch = latent.shape[0]
        projected: torch.Tensor = self.decoder_embed(latent)
        fillers = self.mask_token.expand(batch, self.n_masked, -1)
        full = torch.cat([projected, fillers], dim=1)
        full = gather_tokens(full, ids_restore)
        full = full + self.decoder_pos_embed

        for block in self.decoder_blocks:
            full = block(full)
        out: torch.Tensor = self.decoder_pred(self.decoder_norm(full))
        return out

    def targets(self, x: torch.Tensor) -> torch.Tensor:
        """Patchified reconstruction targets, ``(B, N, P)``."""
        # Buffers are registered at runtime, so the type checker sees them as
        # Tensor | Module. They are registered rather than stored as plain
        # attributes so they follow the model to mps or cuda.
        indices = cast(torch.Tensor, self.target_indices)
        return patchify(x.index_select(1, indices), self.patch)

    def forward(
        self, x: torch.Tensor, generator: torch.Generator | None = None
    ) -> dict[str, torch.Tensor]:
        """Mask, encode, decode and score.

        Args:
            x: ``(B, C, H, W)`` field, time already folded into channels.
            generator: Optional RNG for reproducible masking in tests.

        Returns:
            ``loss`` (scalar), ``prediction`` (B, N, P) and ``mask`` (B, N).

        """
        tokens = self.backbone.embed(x)
        visible, mask, ids_restore = self.random_masking(tokens, generator)
        latent = self.backbone.encode(visible)
        prediction = self.decode(latent, ids_restore)
        weights = cast(torch.Tensor, self.token_weights)
        loss = masked_patch_loss(prediction, self.targets(x), mask, weights)
        return {"loss": loss, "prediction": prediction, "mask": mask}

    @torch.no_grad()
    def reconstruct(
        self, x: torch.Tensor, generator: torch.Generator | None = None
    ) -> dict[str, torch.Tensor]:
        """Fields for visual inspection: truth, masked input, reconstruction.

        Visible patches are passed through unchanged and predictions are used
        only where the model could not see -- which is what you want to look
        at when judging whether pretraining is working.
        """
        out = self.forward(x, generator)
        target = self.targets(x)
        mask = out["mask"].unsqueeze(-1)
        filled = target * (1 - mask) + out["prediction"] * mask

        grid, channels = self.grid, self.n_target_channels
        return {
            "truth": unpatchify(target, self.patch, grid, channels),
            "masked_input": unpatchify(target * (1 - mask), self.patch, grid, channels),
            "reconstruction": unpatchify(filled, self.patch, grid, channels),
            "prediction": unpatchify(out["prediction"], self.patch, grid, channels),
        }
