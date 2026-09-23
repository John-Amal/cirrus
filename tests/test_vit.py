"""Tests for the backbone.

The most informative test here is the mirror image of one in
``test_attention.py``: bare attention *is* permutation-equivariant, and the
backbone must *not* be, because position embeddings have been added. If the
backbone were still equivariant, the position embeddings would not be
reaching the tokens, and the model would be unable to tell the tropics from
the pole -- while training perfectly happily.
"""

from __future__ import annotations

import pytest
import torch

from cirrus.models.vit import BackboneSpec, ViT

CHANNELS = 54  # 2 timesteps x 27 channels
GRID = (32, 64)
BATCH = 2


@pytest.fixture
def backbone() -> ViT:
    torch.manual_seed(0)
    spec = BackboneSpec(dim=64, depth=2, n_heads=4)
    return ViT(in_channels=CHANNELS, spec=spec, grid=GRID).eval()


def field() -> torch.Tensor:
    return torch.randn(BATCH, CHANNELS, *GRID)


def test_output_shape(backbone: ViT):
    assert backbone.n_tokens == 128
    assert backbone(field()).shape == (BATCH, 128, 64)


def test_depth_is_respected(backbone: ViT):
    assert len(backbone.blocks) == 2


def test_forward_is_embed_then_encode(backbone: ViT):
    """The split used by MAE must not change what a full forward computes."""
    x = field()
    torch.testing.assert_close(backbone(x), backbone.encode(backbone.embed(x)))


def test_encoding_is_equivariant_given_fixed_tokens(backbone: ViT):
    """The stack itself carries no position sense; the tokens do."""
    tokens = backbone.embed(field())
    order = torch.randperm(backbone.n_tokens)
    torch.testing.assert_close(
        backbone.encode(tokens[:, order]),
        backbone.encode(tokens)[:, order],
        rtol=1e-4,
        atol=1e-5,
    )


def test_position_embeddings_break_translation_equivariance(backbone: ViT):
    """Rolling the field by one patch must not simply roll the outputs.

    Paired with the test below, this shows the position embeddings are doing
    real work: zero them and the model becomes translation-equivariant, which
    would mean it cannot distinguish the Pacific from the Atlantic.
    """
    x = field()
    rows, cols = backbone.patch_embed.n_patches_lat, backbone.patch_embed.n_patches_lon
    rolled_input = torch.roll(x, shifts=backbone.spec.patch, dims=-1)

    out = backbone(x).reshape(BATCH, rows, cols, -1)
    out_rolled = backbone(rolled_input).reshape(BATCH, rows, cols, -1)
    if_equivariant = torch.roll(out, shifts=1, dims=2)

    assert not torch.allclose(out_rolled, if_equivariant, atol=1e-4)


def test_without_position_embeddings_it_is_equivariant(backbone: ViT):
    """The control: with position embeddings zeroed, rolling commutes."""
    with torch.no_grad():
        backbone.pos_embed.zero_()

    x = field()
    rows, cols = backbone.patch_embed.n_patches_lat, backbone.patch_embed.n_patches_lon
    rolled_input = torch.roll(x, shifts=backbone.spec.patch, dims=-1)

    out = backbone(x).reshape(BATCH, rows, cols, -1)
    out_rolled = backbone(rolled_input).reshape(BATCH, rows, cols, -1)
    if_equivariant = torch.roll(out, shifts=1, dims=2)

    torch.testing.assert_close(out_rolled, if_equivariant, rtol=1e-4, atol=1e-5)


def test_identical_patches_at_different_positions_differ(backbone: ViT):
    """A uniform field still yields position-dependent tokens."""
    uniform = torch.ones(1, CHANNELS, *GRID)
    tokens = backbone.embed(uniform)[0]
    assert not torch.allclose(tokens[0], tokens[50], atol=1e-6)


def test_encode_accepts_a_token_subset(backbone: ViT):
    """MAE encodes only the visible quarter of the sequence."""
    tokens = backbone.embed(field())
    kept = tokens[:, :32]
    assert backbone.encode(kept).shape == (BATCH, 32, 64)


def test_gradients_reach_every_parameter(backbone: ViT):
    backbone(field()).sum().backward()
    for name, param in backbone.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient in {name}"


def test_position_embedding_receives_gradient(backbone: ViT):
    """It is a parameter, so it must actually be learned."""
    backbone(field()).sum().backward()
    assert backbone.pos_embed.grad is not None
    assert backbone.pos_embed.grad.abs().sum() > 0


def test_output_is_finite_and_normalised(backbone: ViT):
    """The final LayerNorm should leave tokens at roughly unit scale."""
    out = backbone(field())
    assert torch.isfinite(out).all()
    assert 0.5 < out.std().item() < 2.0


def test_parameter_count_is_reasonable():
    """A sanity check on model size, not a specification."""
    model = ViT(in_channels=CHANNELS, spec=BackboneSpec(dim=256, depth=6))
    millions = model.n_parameters / 1e6
    assert 1.0 < millions < 50.0, f"unexpected size: {millions:.1f}M parameters"


def test_spec_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("dpeth: 6\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        BackboneSpec.from_yaml(path)


def test_heads_must_divide_dim():
    with pytest.raises(ValueError, match="divisible"):
        ViT(in_channels=CHANNELS, spec=BackboneSpec(dim=64, n_heads=7))
