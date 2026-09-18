"""A deliberately trivial model.

Phase 0 exists to prove the *plumbing* works: config -> data -> model ->
loss -> optimiser -> checkpoint. Using a real model here would confound
"is my training loop correct?" with "is my architecture any good?".

This one maps a gridded field to a gridded field through two convolutions.
It has no scientific content whatsoever and will be deleted in Phase 2.
"""

from __future__ import annotations

import torch
from torch import nn


class DummyModel(nn.Module):
    """Two 3x3 convolutions with a nonlinearity between them.

    Args:
        n_channels: Number of input and output channels (variables).
        hidden_dim: Width of the intermediate representation.

    """

    def __init__(self, n_channels: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, n_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, channels, height, width)`` to the same shape."""
        out: torch.Tensor = self.net(x)
        return out

    @property
    def n_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
