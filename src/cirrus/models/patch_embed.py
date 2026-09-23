"""Turning gridded fields into a sequence of tokens.

A transformer consumes a sequence. A weather state is a grid. The bridge is
patching: cut the globe into a grid of patches and project each one to a
vector. At 5.625 deg the grid is 32x64, so patches of 4x4 give 8x16 = 128
tokens -- small enough that full attention over all pairs is cheap.

**Token order is row-major**: token ``i * n_patches_lon + j`` is the patch at
patch-row ``i`` (south to north, since latitude is ascending) and patch-column
``j``. Everything downstream depends on this, so ``patchify`` and the
convolution here are written to agree, and a test checks they do.

The time dimension is folded into channels: two input steps of 27 channels
become 54 channels of one image. This is what GraphCast and AIFS do. The
alternative -- attending over time as extra tokens -- costs more and buys
little at a history length of two.
"""

from __future__ import annotations

import torch
from torch import nn


def flatten_time(x: torch.Tensor) -> torch.Tensor:
    """Fold a ``(B, T, C, H, W)`` window into ``(B, T*C, H, W)``.

    Channel order is time-major: all channels of step 0, then all of step 1.
    """
    if x.ndim != 5:
        raise ValueError(
            f"expected (batch, time, channel, lat, lon), got {tuple(x.shape)}"
        )
    batch, steps, channels, height, width = x.shape
    return x.reshape(batch, steps * channels, height, width)


def patchify(x: torch.Tensor, patch: int) -> torch.Tensor:
    """Cut ``(B, C, H, W)`` into ``(B, N, C*patch*patch)`` patch vectors.

    Used for the reconstruction target in masked-autoencoder pretraining,
    where the loss is computed per patch. Exactly inverted by
    :func:`unpatchify`.
    """
    if x.ndim != 4:
        raise ValueError(f"expected (batch, channel, lat, lon), got {tuple(x.shape)}")
    batch, channels, height, width = x.shape
    if height % patch or width % patch:
        raise ValueError(f"grid {height}x{width} is not divisible by patch {patch}")

    rows, cols = height // patch, width // patch
    out = x.reshape(batch, channels, rows, patch, cols, patch)
    # (B, C, rows, p, cols, p) -> (B, rows, cols, C, p, p): patch position
    # first, so flattening gives row-major token order.
    out = out.permute(0, 2, 4, 1, 3, 5)
    return out.reshape(batch, rows * cols, channels * patch * patch)


def unpatchify(
    tokens: torch.Tensor, patch: int, grid: tuple[int, int], channels: int
) -> torch.Tensor:
    """Reassemble ``(B, N, C*patch*patch)`` patch vectors into ``(B, C, H, W)``."""
    height, width = grid
    rows, cols = height // patch, width // patch
    batch, n_tokens, width_per_token = tokens.shape
    if n_tokens != rows * cols:
        raise ValueError(f"expected {rows * cols} tokens, got {n_tokens}")
    if width_per_token != channels * patch * patch:
        raise ValueError(
            f"expected token width {channels * patch * patch}, got {width_per_token}"
        )

    out = tokens.reshape(batch, rows, cols, channels, patch, patch)
    out = out.permute(0, 3, 1, 4, 2, 5)
    return out.reshape(batch, channels, height, width)


class PatchEmbed(nn.Module):
    """Project each patch of a gridded field to a token vector.

    Implemented as a convolution whose kernel and stride both equal the patch
    size, which applies one shared linear map per patch and touches each pixel
    exactly once. It is a linear layer, not a feature extractor: the weight
    sharing means the same projection is used everywhere, so position
    information has to come from the embeddings added in the backbone.

    Args:
        in_channels: Channels per timestep times number of timesteps.
        dim: Token width.
        patch: Patch edge in grid cells.
        grid: ``(lat, lon)`` size of the input field.

    """

    def __init__(
        self,
        in_channels: int,
        dim: int = 256,
        patch: int = 4,
        grid: tuple[int, int] = (32, 64),
    ) -> None:
        super().__init__()
        height, width = grid
        if height % patch or width % patch:
            raise ValueError(f"grid {height}x{width} is not divisible by patch {patch}")

        self.in_channels = in_channels
        self.dim = dim
        self.patch = patch
        self.grid = grid
        self.n_patches_lat = height // patch
        self.n_patches_lon = width // patch
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch, stride=patch)

    @property
    def n_tokens(self) -> int:
        """Number of tokens produced per field."""
        return self.n_patches_lat * self.n_patches_lon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, H, W)`` to ``(B, N, dim)`` in row-major token order."""
        if x.ndim != 4:
            raise ValueError(
                f"expected (batch, channel, lat, lon), got {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {x.shape[1]}")
        if x.shape[-2:] != self.grid:
            raise ValueError(f"expected grid {self.grid}, got {tuple(x.shape[-2:])}")

        projected: torch.Tensor = self.proj(x)  # (B, dim, rows, cols)
        return projected.flatten(2).transpose(1, 2)
