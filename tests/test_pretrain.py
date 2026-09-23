"""Tests for the pretraining helpers.

The loop itself is exercised by running it; these cover the three pieces that
are easy to get wrong and impossible to notice: the learning-rate schedule,
which parameters escape weight decay, and which channels are reconstruction
targets after the time dimension is folded into channels.
"""

from __future__ import annotations

import pytest
import torch

from cirrus.models.mae import MaeSpec, MaskedAutoencoder
from cirrus.models.vit import BackboneSpec, ViT
from cirrus.train.pretrain import (
    PretrainSpec,
    dynamic_target_indices,
    learning_rate_at,
    parameter_groups,
)

BASE_LR = 1e-3
TOTAL = 1000
WARMUP = 100


def test_warmup_rises_linearly_from_near_zero():
    assert learning_rate_at(0, TOTAL, BASE_LR, WARMUP) == pytest.approx(
        BASE_LR / WARMUP
    )
    half = learning_rate_at(WARMUP // 2 - 1, TOTAL, BASE_LR, WARMUP)
    assert half == pytest.approx(BASE_LR / 2, rel=1e-6)


def test_peak_is_reached_at_the_end_of_warmup():
    assert learning_rate_at(WARMUP - 1, TOTAL, BASE_LR, WARMUP) == pytest.approx(
        BASE_LR
    )


def test_cosine_decays_monotonically_after_warmup():
    values = [learning_rate_at(s, TOTAL, BASE_LR, WARMUP) for s in range(WARMUP, TOTAL)]
    assert all(b <= a for a, b in zip(values, values[1:], strict=False))


def test_final_learning_rate_is_the_floor():
    final = learning_rate_at(TOTAL, TOTAL, BASE_LR, WARMUP, min_lr_ratio=0.01)
    assert final == pytest.approx(BASE_LR * 0.01, rel=1e-6)


def test_schedule_never_exceeds_the_peak():
    values = [learning_rate_at(s, TOTAL, BASE_LR, WARMUP) for s in range(TOTAL + 50)]
    assert max(values) <= BASE_LR * (1 + 1e-9)
    assert min(values) > 0


def test_no_warmup_starts_at_the_peak():
    assert learning_rate_at(0, TOTAL, BASE_LR, warmup_steps=0) == pytest.approx(BASE_LR)


def model() -> MaskedAutoencoder:
    torch.manual_seed(0)
    backbone = ViT(
        in_channels=8, spec=BackboneSpec(dim=32, depth=1, n_heads=4), grid=(16, 32)
    )
    return MaskedAutoencoder(backbone, [0, 1], MaeSpec(latitude_weighted=False))


def test_parameter_groups_cover_every_parameter():
    mae = model()
    groups = parameter_groups(mae, weight_decay=0.05)
    counted = sum(p.numel() for group in groups for p in group["params"])
    assert counted == sum(p.numel() for p in mae.parameters() if p.requires_grad)


def test_norms_biases_and_embeddings_escape_weight_decay():
    """Decaying a LayerNorm gain fights the normalisation it provides."""
    mae = model()
    decayed, undecayed = parameter_groups(mae, 0.05)
    assert decayed["weight_decay"] == 0.05
    assert undecayed["weight_decay"] == 0.0

    exempt = {id(p) for p in undecayed["params"]}
    for name, param in mae.named_parameters():
        if param.ndim < 2 or "pos_embed" in name or "mask_token" in name:
            assert id(param) in exempt, f"{name} should not be decayed"
        else:
            assert id(param) not in exempt, f"{name} should be decayed"


def test_target_indices_follow_time_major_layout():
    """Two steps of 27 channels, the first 21 of each being dynamic."""
    indices = dynamic_target_indices(n_steps=2, n_channels_per_step=27, n_dynamic=21)
    assert len(indices) == 42
    assert indices[:3] == [0, 1, 2]
    assert indices[20] == 20  # last dynamic channel of step 0
    assert indices[21] == 27  # first dynamic channel of step 1
    assert max(indices) == 47
    assert 21 not in indices  # a static channel of step 0


def test_target_indices_exclude_statics_and_forcings():
    indices = set(dynamic_target_indices(2, 27, 21))
    statics_and_forcings = set(range(21, 27)) | set(range(48, 54))
    assert not (indices & statics_and_forcings)


def test_spec_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("epocs: 3\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        PretrainSpec.from_yaml(path)
