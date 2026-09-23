"""The backbone: a vision transformer over patches of the globe.

Structure is deliberately plain -- patch embedding, position embeddings, a
stack of pre-norm transformer blocks, a final norm. Nothing clever. The
research question in this project is about the training objective and the
treatment of the tail, not the architecture, so the architecture should be
the boring, well-understood baseline that isolates those variables.

``embed`` and ``encode`` are separate methods rather than one ``forward``
because masked-autoencoder pretraining needs to intervene between them: embed
every patch, add position information, discard most of the tokens, and encode
only the survivors. That is what makes MAE pretraining cheap -- the encoder
sees a quarter of the sequence, and attention cost grows with the square of
sequence length.

Position embeddings are learned. They matter: attention itself is
permutation-equivariant, so without them the model cannot tell the tropics
from the pole. A fixed sinusoidal scheme periodic in longitude is the natural
alternative here, and would compose more neatly with the longitude-roll
augmentation -- a worthwhile Phase 4 ablation rather than a decision to make
blind now.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import nn

from cirrus.config import read_yaml_mapping
from cirrus.models.attention import TransformerBlock
from cirrus.models.patch_embed import PatchEmbed


@dataclass(frozen=True)
class BackboneSpec:
    """Architecture of the backbone."""

    dim: int = 256
    depth: int = 6
    n_heads: int = 8
    patch: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> BackboneSpec:
        """Load a spec, rejecting unknown keys."""
        raw: dict[str, Any] = read_yaml_mapping(path, (f.name for f in fields(cls)))
        return cls(**raw)


class ViT(nn.Module):
    """Patch-based transformer encoder over a gridded field.

    Args:
        in_channels: Channels of the input field, time already folded in.
        spec: Architecture settings.
        grid: ``(lat, lon)`` size of the input field.

    """

    def __init__(
        self,
        in_channels: int,
        spec: BackboneSpec | None = None,
        grid: tuple[int, int] = (32, 64),
    ) -> None:
        super().__init__()
        self.spec = spec or BackboneSpec()
        self.grid = grid
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            dim=self.spec.dim,
            patch=self.spec.patch,
            grid=grid,
        )
        # One learned vector per patch position, added to the token that lands
        # there. Small init (std 0.02) so early training is dominated by the
        # content of the patch rather than by where it is.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.n_tokens, self.spec.dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=self.spec.dim,
                    n_heads=self.spec.n_heads,
                    hidden_ratio=self.spec.mlp_ratio,
                    dropout=self.spec.dropout,
                )
                for _ in range(self.spec.depth)
            ]
        )
        self.norm = nn.LayerNorm(self.spec.dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Truncated-normal linear weights, zero bias, standard LayerNorm."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @property
    def n_tokens(self) -> int:
        """Tokens produced for a full (unmasked) field."""
        return self.patch_embed.n_tokens

    @property
    def n_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Patch-embed ``(B, C, H, W)`` and add position embeddings."""
        tokens: torch.Tensor = self.patch_embed(x)
        return tokens + self.pos_embed

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run the transformer stack over ``(B, N, dim)`` tokens.

        Any number of tokens is accepted, not just a full field: MAE passes a
        subset here. Position information is already baked into the tokens by
        :meth:`embed`, so dropping tokens loses nothing the encoder needs.
        """
        for block in self.blocks:
            tokens = block(tokens)
        out: torch.Tensor = self.norm(tokens)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a full field: ``(B, C, H, W)`` to ``(B, N, dim)``."""
        return self.encode(self.embed(x))
