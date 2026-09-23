"""Tests for masked-autoencoder pretraining.

Two failure modes here are silent and expensive. If ``ids_restore`` is wrong,
the decoder puts tokens back in the wrong places and every prediction is
scored against a different patch's target -- training proceeds, to a worse
model. If the loss accidentally includes visible patches, the model earns
most of its reward by copying what it can already see, and the loss curve
looks excellent while it learns much less.
"""

from __future__ import annotations

import pytest
import torch

from cirrus.models.mae import (
    MaeSpec,
    MaskedAutoencoder,
    gather_tokens,
    latitude_token_weights,
    masked_patch_loss,
)
from cirrus.models.vit import BackboneSpec, ViT

CHANNELS = 12
DYNAMIC = list(range(8))  # the rest stand in for statics and forcings
GRID = (32, 64)
BATCH = 2
LATITUDES = torch.linspace(-87.1875, 87.1875, GRID[0])


@pytest.fixture
def mae() -> MaskedAutoencoder:
    torch.manual_seed(0)
    backbone = ViT(
        in_channels=CHANNELS,
        spec=BackboneSpec(dim=64, depth=2, n_heads=4, patch=4),
        grid=GRID,
    )
    return MaskedAutoencoder(backbone, DYNAMIC, MaeSpec(), LATITUDES).eval()


def field() -> torch.Tensor:
    return torch.randn(BATCH, CHANNELS, *GRID)


def test_masking_counts(mae: MaskedAutoencoder):
    assert mae.n_tokens == 128
    assert mae.n_masked == 96  # 75% of 128
    assert mae.n_visible == 32


def test_mask_marks_exactly_the_hidden_tokens(mae: MaskedAutoencoder):
    tokens = mae.backbone.embed(field())
    visible, mask, _ = mae.random_masking(tokens)
    assert visible.shape == (BATCH, mae.n_visible, 64)
    assert mask.shape == (BATCH, mae.n_tokens)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    torch.testing.assert_close(
        mask.sum(dim=1), torch.full((BATCH,), float(mae.n_masked))
    )


def test_ids_restore_is_a_true_inverse(mae: MaskedAutoencoder):
    """Shuffle then restore must return tokens to their original order."""
    tokens = mae.backbone.embed(field())
    batch, n_tokens, _ = tokens.shape
    noise = torch.rand(batch, n_tokens)
    ids_shuffle = noise.argsort(dim=1)
    ids_restore = ids_shuffle.argsort(dim=1)

    shuffled = gather_tokens(tokens, ids_shuffle)
    restored = gather_tokens(shuffled, ids_restore)
    torch.testing.assert_close(restored, tokens)


def test_each_sample_gets_its_own_mask(mae: MaskedAutoencoder):
    """A shared mask across the batch would waste most of the signal."""
    torch.manual_seed(1)
    tokens = mae.backbone.embed(torch.randn(8, CHANNELS, *GRID))
    _, mask, _ = mae.random_masking(tokens)
    assert not torch.equal(mask[0], mask[1])


def test_masking_is_reproducible_with_a_generator(mae: MaskedAutoencoder):
    tokens = mae.backbone.embed(field())

    def draw() -> torch.Tensor:
        generator = torch.Generator().manual_seed(7)
        return mae.random_masking(tokens, generator)[1]

    torch.testing.assert_close(draw(), draw())


def test_masking_works_with_a_cpu_generator_on_any_device(mae: MaskedAutoencoder):
    """Regression: a CPU generator must not force the tokens onto CPU.

    Training crashed here on mps -- torch.rand refuses a generator whose
    device differs from the tensor's. The noise is now drawn on CPU and
    moved, which also makes masks identical across devices.
    """
    tokens = mae.backbone.embed(field())
    generator = torch.Generator().manual_seed(3)
    _, mask, _ = mae.random_masking(tokens, generator)
    assert mask.device == tokens.device
    assert mask.sum().item() == pytest.approx(BATCH * mae.n_masked)


def test_loss_ignores_visible_patches():
    """Changing predictions on visible patches must not change the loss."""
    torch.manual_seed(0)
    prediction = torch.randn(2, 10, 16)
    target = torch.randn(2, 10, 16)
    mask = torch.zeros(2, 10)
    mask[:, :4] = 1.0

    before = masked_patch_loss(prediction, target, mask)
    prediction[:, 4:] += 100.0  # visible patches, wildly wrong
    after = masked_patch_loss(prediction, target, mask)
    torch.testing.assert_close(before, after)


def test_loss_is_zero_for_a_perfect_reconstruction():
    target = torch.randn(2, 10, 16)
    mask = torch.ones(2, 10)
    assert masked_patch_loss(target.clone(), target, mask).item() == pytest.approx(0.0)


def test_latitude_weights_favour_the_equator():
    weights = latitude_token_weights(LATITUDES, patch=4, n_patches_lon=16)
    assert weights.shape == (128,)
    assert weights.mean().item() == pytest.approx(1.0, abs=1e-5)
    equator_row = weights.reshape(8, 16)[4]
    pole_row = weights.reshape(8, 16)[0]
    assert equator_row.mean() > 3 * pole_row.mean()


def test_latitude_weighting_changes_the_loss():
    torch.manual_seed(0)
    prediction, target = torch.randn(1, 128, 16), torch.randn(1, 128, 16)
    mask = torch.ones(1, 128)
    weights = latitude_token_weights(LATITUDES, patch=4, n_patches_lon=16)
    assert not torch.allclose(
        masked_patch_loss(prediction, target, mask),
        masked_patch_loss(prediction, target, mask, weights),
    )


def test_forward_returns_loss_prediction_and_mask(mae: MaskedAutoencoder):
    out = mae(field())
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])
    assert out["prediction"].shape == (BATCH, 128, len(DYNAMIC) * 16)
    assert out["mask"].shape == (BATCH, 128)


def test_targets_use_only_the_named_channels(mae: MaskedAutoencoder):
    """Statics and forcings must not be reconstruction targets."""
    x = field()
    targets = mae.targets(x)
    assert targets.shape == (BATCH, 128, len(DYNAMIC) * 16)
    # Changing an excluded channel must not change the targets at all.
    x_changed = x.clone()
    x_changed[:, len(DYNAMIC) :] += 5.0
    torch.testing.assert_close(mae.targets(x_changed), targets)


def test_gradients_reach_encoder_and_decoder(mae: MaskedAutoencoder):
    mae(field())["loss"].backward()
    for name, param in mae.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient in {name}"
    assert mae.mask_token.grad.abs().sum() > 0


def test_can_overfit_a_single_batch():
    """The loop-and-objective sanity check, at model scale.

    A model with enough capacity must drive the reconstruction loss on one
    fixed batch far down. If it cannot, the objective or the wiring is broken
    and no amount of data will help.
    """
    torch.manual_seed(0)
    backbone = ViT(
        in_channels=4,
        spec=BackboneSpec(dim=64, depth=2, n_heads=4, patch=8),
        grid=(16, 16),
    )
    model = MaskedAutoencoder(
        backbone,
        [0, 1],
        MaeSpec(mask_ratio=0.5, decoder_depth=1, latitude_weighted=False),
    )
    x = torch.randn(2, 4, 16, 16)
    generator = torch.Generator().manual_seed(0)

    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(300):
        generator.manual_seed(0)  # same mask every step, so the task is fixed
        loss = model(x, generator)["loss"]
        first = loss.item() if first is None else first
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    assert loss.item() < 0.2 * first


def test_reconstruct_returns_fields(mae: MaskedAutoencoder):
    out = mae.reconstruct(field())
    for key in ("truth", "masked_input", "reconstruction", "prediction"):
        assert out[key].shape == (BATCH, len(DYNAMIC), *GRID)
    # Visible patches are passed through untouched.
    assert torch.isfinite(out["reconstruction"]).all()


def test_mask_ratio_must_be_a_proper_fraction():
    with pytest.raises(ValueError, match="mask_ratio"):
        MaeSpec(mask_ratio=1.0)
    with pytest.raises(ValueError, match="mask_ratio"):
        MaeSpec(mask_ratio=0.0)


def test_latitudes_required_when_weighting(mae: MaskedAutoencoder):
    backbone = ViT(in_channels=CHANNELS, spec=BackboneSpec(dim=64, depth=1), grid=GRID)
    with pytest.raises(ValueError, match="latitudes"):
        MaskedAutoencoder(backbone, DYNAMIC, MaeSpec(latitude_weighted=True), None)


def test_target_indices_are_validated():
    backbone = ViT(in_channels=CHANNELS, spec=BackboneSpec(dim=64, depth=1), grid=GRID)
    spec = MaeSpec(latitude_weighted=False)
    with pytest.raises(ValueError, match="at least one channel"):
        MaskedAutoencoder(backbone, [], spec)
    with pytest.raises(ValueError, match="out of range"):
        MaskedAutoencoder(backbone, [0, 99], spec)
