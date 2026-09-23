"""Tests for the transformer components.

The hand-written attention is checked against PyTorch's fused
``scaled_dot_product_attention``: if the two disagree, the hand-written one
is wrong, and no amount of training-curve staring would tell you that.

Token ordering gets its own test. ``patchify`` and the patch-embedding
convolution must agree on which token is which patch, or the
masked-autoencoder loss in Phase 2 compares predictions to the wrong targets
-- and still trains, just to a worse model.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from cirrus.models.attention import Mlp, MultiHeadSelfAttention, TransformerBlock
from cirrus.models.patch_embed import (
    PatchEmbed,
    flatten_time,
    patchify,
    unpatchify,
)

DIM = 32
HEADS = 4
TOKENS = 10
BATCH = 3


@pytest.fixture
def attn() -> MultiHeadSelfAttention:
    torch.manual_seed(0)
    return MultiHeadSelfAttention(DIM, HEADS).eval()


def test_attention_preserves_shape(attn: MultiHeadSelfAttention):
    x = torch.randn(BATCH, TOKENS, DIM)
    assert attn(x).shape == x.shape


def test_matches_pytorch_reference(attn: MultiHeadSelfAttention):
    """The hand-written implementation must equal the fused one."""
    x = torch.randn(BATCH, TOKENS, DIM)

    qkv = attn.qkv(x).reshape(BATCH, TOKENS, 3, HEADS, DIM // HEADS)
    query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    reference = F.scaled_dot_product_attention(query, key, value)
    reference = reference.transpose(1, 2).reshape(BATCH, TOKENS, DIM)
    reference = attn.proj(reference)

    torch.testing.assert_close(attn(x), reference, rtol=1e-5, atol=1e-6)


def test_attention_weights_are_distributions(attn: MultiHeadSelfAttention):
    weights = attn.attention_weights(torch.randn(BATCH, TOKENS, DIM))
    assert weights.shape == (BATCH, HEADS, TOKENS, TOKENS)
    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones(BATCH, HEADS, TOKENS), rtol=1e-5, atol=1e-6
    )
    assert (weights >= 0).all()


def test_attention_is_permutation_equivariant(attn: MultiHeadSelfAttention):
    """Shuffling tokens shuffles outputs identically.

    Attention alone carries no notion of position -- which is precisely why
    position embeddings must be added in the backbone.
    """
    x = torch.randn(BATCH, TOKENS, DIM)
    order = torch.randperm(TOKENS)
    torch.testing.assert_close(
        attn(x[:, order]), attn(x)[:, order], rtol=1e-5, atol=1e-6
    )


def test_single_token_returns_its_own_value(attn: MultiHeadSelfAttention):
    """With one token the softmax is 1, so the output is proj(value)."""
    x = torch.randn(BATCH, 1, DIM)
    qkv = attn.qkv(x).reshape(BATCH, 1, 3, HEADS, DIM // HEADS)
    value = qkv.permute(2, 0, 3, 1, 4).unbind(0)[2]
    expected = attn.proj(value.transpose(1, 2).reshape(BATCH, 1, DIM))
    torch.testing.assert_close(attn(x), expected, rtol=1e-5, atol=1e-6)


def test_gradients_reach_every_parameter(attn: MultiHeadSelfAttention):
    attn(torch.randn(BATCH, TOKENS, DIM)).sum().backward()
    for name, param in attn.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient in {name}"


def test_heads_must_divide_dim():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadSelfAttention(dim=30, n_heads=4)


def test_rejects_wrong_rank(attn: MultiHeadSelfAttention):
    with pytest.raises(ValueError, match="batch, tokens, dim"):
        attn(torch.randn(BATCH, DIM))


def test_block_preserves_shape_and_is_residual():
    """A block returns its input plus learned updates, so shape is unchanged."""
    torch.manual_seed(0)
    block = TransformerBlock(DIM, HEADS).eval()
    x = torch.randn(BATCH, TOKENS, DIM)
    assert block(x).shape == x.shape
    # Zero the output projections: the residual path alone must pass x through.
    with torch.no_grad():
        block.attn.proj.weight.zero_()
        block.attn.proj.bias.zero_()
        block.mlp.fc2.weight.zero_()
        block.mlp.fc2.bias.zero_()
    torch.testing.assert_close(block(x), x, rtol=1e-6, atol=1e-6)


def test_mlp_is_token_wise():
    """Each token is processed independently of the others."""
    torch.manual_seed(0)
    mlp = Mlp(DIM).eval()
    x = torch.randn(BATCH, TOKENS, DIM)
    full = mlp(x)
    single = mlp(x[:, 2:3])
    torch.testing.assert_close(full[:, 2:3], single, rtol=1e-5, atol=1e-6)


# --- patch embedding -------------------------------------------------------


def test_token_count_and_shape():
    embed = PatchEmbed(in_channels=6, dim=DIM, patch=4, grid=(32, 64))
    assert embed.n_tokens == 8 * 16 == 128
    out = embed(torch.randn(BATCH, 6, 32, 64))
    assert out.shape == (BATCH, 128, DIM)


def test_patch_must_divide_grid():
    with pytest.raises(ValueError, match="divisible"):
        PatchEmbed(in_channels=1, patch=5, grid=(32, 64))


def test_rejects_wrong_channels_or_grid():
    embed = PatchEmbed(in_channels=6, dim=DIM, patch=4, grid=(32, 64))
    with pytest.raises(ValueError, match="channels"):
        embed(torch.randn(BATCH, 5, 32, 64))
    with pytest.raises(ValueError, match="grid"):
        embed(torch.randn(BATCH, 6, 16, 64))


def test_patchify_unpatchify_round_trip():
    x = torch.randn(BATCH, 5, 32, 64)
    tokens = patchify(x, patch=4)
    assert tokens.shape == (BATCH, 128, 5 * 4 * 4)
    torch.testing.assert_close(unpatchify(tokens, 4, (32, 64), 5), x)


def test_patchify_and_convolution_agree_on_token_order():
    """The critical one: both must call the same patch token number i.

    A field is built where every patch is a distinct constant. patchify must
    return those constants in the same order the convolution does; if they
    disagree, the MAE loss silently compares each prediction to a different
    patch's target.
    """
    patch, rows, cols = 4, 8, 16
    x = torch.zeros(1, 1, rows * patch, cols * patch)
    for i in range(rows):
        for j in range(cols):
            x[0, 0, i * patch : (i + 1) * patch, j * patch : (j + 1) * patch] = (
                i * cols + j
            )

    from_patchify = patchify(x, patch)[0, :, 0]

    embed = PatchEmbed(
        in_channels=1, dim=1, patch=patch, grid=(rows * patch, cols * patch)
    )
    with torch.no_grad():  # make the conv compute a plain patch mean
        embed.proj.weight.fill_(1.0 / (patch * patch))
        embed.proj.bias.zero_()
    from_conv = embed(x)[0, :, 0]

    torch.testing.assert_close(from_patchify, from_conv, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        from_conv, torch.arange(rows * cols, dtype=torch.float32)
    )


def test_flatten_time_is_time_major():
    """Channels of step 0 come first, then channels of step 1."""
    x = torch.randn(BATCH, 2, 7, 32, 64)
    flat = flatten_time(x)
    assert flat.shape == (BATCH, 14, 32, 64)
    torch.testing.assert_close(flat[:, :7], x[:, 0])
    torch.testing.assert_close(flat[:, 7:], x[:, 1])


def test_flatten_time_rejects_wrong_rank():
    with pytest.raises(ValueError, match="batch, time, channel"):
        flatten_time(torch.randn(BATCH, 7, 32, 64))
