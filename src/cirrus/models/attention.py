"""Multi-head self-attention, written out.

``F.scaled_dot_product_attention`` would do this in one call, and is faster
because it fuses the operations. It is used here only as a reference to test
against: the point of writing it out is that you can then debug it, modify
the objective, and explain every line of it.

The mechanism, in one paragraph. Every token emits three vectors: a **query**
(what am I looking for), a **key** (what do I offer), and a **value** (what I
will pass on). The score between two tokens is the dot product of one's query
with the other's key, so tokens whose queries and keys align attend to each
other. Softmax over the scores turns them into weights that sum to one, and
each token's output is the weighted sum of all values. Nothing in this refers
to position -- attention is permutation-equivariant, which is exactly why
position embeddings have to be added separately.

For gridded weather fields, a token is a patch of the globe, and attention
lets any patch read from any other in a single layer. That is the property a
convolution lacks: teleconnections between distant regions do not have to be
propagated through many layers of local receptive fields.
"""

from __future__ import annotations

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Self-attention over a sequence of tokens.

    Args:
        dim: Token width. Must divide evenly by ``n_heads``.
        n_heads: Number of attention heads. Each head works in a
            ``dim // n_heads`` subspace, so several relationships can be
            represented at once without increasing cost.
        dropout: Dropout applied to the attention weights during training.
        bias: Whether the projections carry bias terms.

    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by n_heads {n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # 1/sqrt(head_dim): dot products of two random vectors of length d have
        # standard deviation sqrt(d). Without this the scores grow with width,
        # softmax saturates, and gradients vanish before training starts.
        self.scale = self.head_dim**-0.5

        # One projection producing Q, K and V together: three matmuls of the
        # same input fused into one, which is how every transformer does it.
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def _attend(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the attended output and the attention weights.

        Shapes, with B batch, N tokens, D width, H heads, d head width:
        ``x`` is (B, N, D); the returned output is (B, N, D) and the weights
        are (B, H, N, N).
        """
        if x.ndim != 3:
            raise ValueError(f"expected (batch, tokens, dim), got {tuple(x.shape)}")
        batch, tokens, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"expected token width {self.dim}, got {dim}")

        # (B, N, 3D) -> (3, B, H, N, d): separate Q/K/V, then move heads next
        # to the batch so every head is an independent attention problem.
        qkv: torch.Tensor = self.qkv(x)
        qkv = qkv.reshape(batch, tokens, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        scores: torch.Tensor = (query @ key.transpose(-2, -1)) * self.scale
        weights: torch.Tensor = scores.softmax(dim=-1)
        attended: torch.Tensor = self.attn_dropout(weights) @ value

        # (B, H, N, d) -> (B, N, D): put heads back beside their tokens before
        # merging, or the head outputs are interleaved into the wrong columns.
        merged: torch.Tensor = attended.transpose(1, 2).reshape(batch, tokens, dim)
        out: torch.Tensor = self.proj_dropout(self.proj(merged))
        return out, weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over ``(batch, tokens, dim)``, returning the same shape."""
        out, _ = self._attend(x)
        return out

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Attention weights, ``(batch, heads, tokens, tokens)``.

        For inspection and tests only -- not used in training. Each row sums
        to one: it is a distribution over which tokens a token reads from.
        """
        _, weights = self._attend(x)
        return weights


class Mlp(nn.Module):
    """The position-wise feed-forward network of a transformer block.

    Applied to each token independently. Attention moves information between
    tokens; this is where a token processes what it collected.
    """

    def __init__(self, dim: int, hidden_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * hidden_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network token-wise."""
        hidden: torch.Tensor = self.dropout(self.act(self.fc1(x)))
        out: torch.Tensor = self.dropout(self.fc2(hidden))
        return out


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: x + attn(norm(x)), then x + mlp(norm(x)).

    Pre-norm rather than post-norm (the original 2017 arrangement) because it
    trains stably without a learning-rate warmup schedule. The residual path
    stays unnormalised from input to output, so gradients reach early layers
    directly -- which is what makes deep stacks trainable at all.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        hidden_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run attention and feed-forward, each with a residual connection."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
