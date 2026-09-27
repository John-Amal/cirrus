"""Tests for the scoring rules and the precipitation heads.

The two that matter:

``test_crps_matches_the_closed_form`` checks the sample estimator against the
exact Gaussian CRPS. Everything downstream rests on this being right, and an
estimator that is subtly wrong would still train, just to the wrong target.

``test_recovers_a_known_distribution`` fits a censored shifted gamma to data
drawn from a known one. It is the distributional version of the
overfit-one-batch test: if the objective cannot recover parameters that
generated their own data, nothing built on it is worth running.
"""

from __future__ import annotations

import math

import pytest
import torch

from cirrus.losses.crps import crps_sample, mean_score, threshold_weighted_crps
from cirrus.models.heads.csgd import MIN_PARAM, Csgd, CsgdHead, PointHead

GRID = (16, 32)
DIM = 32
PATCH = 4
BATCH = 2


def gaussian_crps(mu: float, sigma: float, y: float) -> float:
    """Closed-form CRPS for a normal predictive distribution."""
    z = (y - mu) / sigma
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def test_crps_matches_the_closed_form():
    torch.manual_seed(0)
    mu, sigma, y = 2.0, 1.5, 3.2
    samples = torch.normal(mu, sigma, size=(20_000, 30))
    target = torch.full((20_000,), y)
    estimate = crps_sample(samples, target).mean().item()
    assert estimate == pytest.approx(gaussian_crps(mu, sigma, y), abs=0.01)


def test_fair_estimator_is_lower_than_the_biased_one():
    """The biased version rewards narrowing the spread; we use the fair one."""
    torch.manual_seed(0)
    samples = torch.normal(2.0, 1.5, size=(5_000, 10))
    target = torch.full((5_000,), 3.2)
    fair = crps_sample(samples, target, fair=True).mean()
    biased = crps_sample(samples, target, fair=False).mean()
    assert fair < biased


def test_crps_is_proper():
    """The true distribution must score better than any misspecification."""
    torch.manual_seed(0)
    mu, sigma, n, m = 2.0, 1.5, 20_000, 30
    truth = torch.normal(mu, sigma, size=(n,))

    def score(mean: float, spread: float) -> float:
        draws = torch.normal(mean, spread, size=(n, m))
        return crps_sample(draws, truth).mean().item()

    correct = score(mu, sigma)
    assert correct < score(mu + 1.0, sigma)  # biased
    assert correct < score(mu, sigma / 2)  # over-confident
    assert correct < score(mu, sigma * 2)  # vague


def test_perfect_point_prediction_scores_zero():
    target = torch.tensor([3.0, 0.0, 7.5])
    samples = target.unsqueeze(-1).expand(-1, 8)
    assert crps_sample(samples, target).abs().max().item() == pytest.approx(
        0.0, abs=1e-6
    )


def test_threshold_weighting_ignores_differences_below_the_threshold():
    """Two forecasts identical above u must score identically under twCRPS."""
    torch.manual_seed(0)
    target = torch.tensor([12.0, 0.5, 30.0])
    threshold = torch.tensor([10.0, 10.0, 10.0])
    above = torch.tensor([[11.0, 13.0], [11.0, 13.0], [28.0, 33.0]])
    below_a = torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    below_b = torch.tensor([[7.0, 8.0], [7.0, 8.0], [7.0, 8.0]])

    first = threshold_weighted_crps(torch.cat([above, below_a], -1), target, threshold)
    second = threshold_weighted_crps(torch.cat([above, below_b], -1), target, threshold)
    torch.testing.assert_close(first, second)


def test_threshold_weighting_still_punishes_a_weak_tail():
    torch.manual_seed(0)
    target = torch.full((4_000,), 25.0)
    threshold = torch.full((4_000,), 10.0)
    strong = torch.normal(25.0, 4.0, size=(4_000, 20))
    weak = torch.normal(14.0, 4.0, size=(4_000, 20))
    assert (
        threshold_weighted_crps(strong, target, threshold).mean()
        < threshold_weighted_crps(weak, target, threshold).mean()
    )


def test_threshold_broadcasts_from_a_grid():
    """Regression: thresholds are (H, W) while targets are (B, H, W)."""
    torch.manual_seed(0)
    samples = torch.rand(2, 4, 8, 6) * 20
    target = torch.rand(2, 4, 8) * 20
    grid_threshold = torch.full((4, 8), 5.0)
    per_sample_threshold = grid_threshold.expand(2, 4, 8)
    torch.testing.assert_close(
        threshold_weighted_crps(samples, target, grid_threshold),
        threshold_weighted_crps(samples, target, per_sample_threshold),
    )


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="sample dimension"):
        crps_sample(torch.randn(4, 8), torch.randn(5))


def test_mean_score_applies_weights():
    score = torch.tensor([[1.0, 3.0]])
    weights = torch.tensor([1.0, 3.0])
    assert mean_score(score).item() == pytest.approx(2.0)
    assert mean_score(score, weights).item() == pytest.approx((1 + 9) / 4)


# --- the distribution -------------------------------------------------------


def test_samples_are_non_negative_and_right_shaped():
    torch.manual_seed(0)
    csgd = Csgd(
        shape=torch.full((3, 4), 2.0),
        scale=torch.full((3, 4), 1.5),
        shift=torch.full((3, 4), -1.0),
    )
    draws = csgd.sample(16)
    assert draws.shape == (3, 4, 16)
    assert (draws >= 0).all()


def test_a_more_negative_shift_makes_it_drier():
    torch.manual_seed(0)

    def dry_fraction(shift: float) -> float:
        csgd = Csgd(
            shape=torch.full((2_000,), 2.0),
            scale=torch.full((2_000,), 1.0),
            shift=torch.full((2_000,), shift),
        )
        return csgd.probability_of_zero(50).mean().item()

    assert dry_fraction(-0.2) < dry_fraction(-1.0) < dry_fraction(-4.0)


def test_gradients_reach_all_three_parameters():
    """The whole reason for the sample estimator: igamma has no shape gradient."""
    torch.manual_seed(0)
    raw = torch.zeros(3, requires_grad=True)
    csgd = Csgd(
        shape=torch.nn.functional.softplus(raw[0]).expand(64),
        scale=torch.nn.functional.softplus(raw[1]).expand(64),
        shift=-torch.nn.functional.softplus(raw[2]).expand(64),
    )
    target = torch.rand(64) * 5
    crps_sample(csgd.sample(16), target).mean().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert (raw.grad.abs() > 0).all(), "a parameter received no gradient"


def test_recovers_a_known_distribution():
    """Fit a CSGD to data from a known CSGD and recover its behaviour.

    Parameters are checked through the quantities that matter -- dry
    fraction, mean and an upper quantile -- rather than one by one, because
    shape and scale trade off against each other and exact recovery of the
    triple is not identifiable from a finite sample.
    """
    torch.manual_seed(0)
    truth = Csgd(
        shape=torch.tensor(2.5), scale=torch.tensor(1.2), shift=torch.tensor(-1.0)
    )
    observations = truth.sample(20_000).reshape(-1)

    raw = torch.zeros(3, requires_grad=True)
    optimiser = torch.optim.Adam([raw], lr=0.05)
    softplus = torch.nn.functional.softplus
    for _ in range(400):
        batch = observations[torch.randint(0, observations.numel(), (512,))]
        fitted = Csgd(
            shape=(softplus(raw[0]) + MIN_PARAM).expand(512),
            scale=(softplus(raw[1]) + MIN_PARAM).expand(512),
            shift=(-softplus(raw[2])).expand(512),
        )
        loss = crps_sample(fitted.sample(24), batch).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    recovered = (
        Csgd(
            shape=(softplus(raw[0]) + MIN_PARAM).detach().expand(20_000),
            scale=(softplus(raw[1]) + MIN_PARAM).detach().expand(20_000),
            shift=(-softplus(raw[2])).detach().expand(20_000),
        )
        .sample(1)
        .reshape(-1)
    )

    assert recovered.mean().item() == pytest.approx(observations.mean().item(), rel=0.2)
    assert (recovered == 0).float().mean().item() == pytest.approx(
        (observations == 0).float().mean().item(), abs=0.1
    )
    assert torch.quantile(recovered, 0.95).item() == pytest.approx(
        torch.quantile(observations, 0.95).item(), rel=0.25
    )


# --- the heads --------------------------------------------------------------


def tokens() -> torch.Tensor:
    n_tokens = (GRID[0] // PATCH) * (GRID[1] // PATCH)
    return torch.randn(BATCH, n_tokens, DIM)


def test_point_head_is_non_negative():
    """Precipitation cannot be negative, and the baselines must not cheat."""
    torch.manual_seed(0)
    head = PointHead(DIM, PATCH, GRID)
    out = head(tokens())
    assert out.shape == (BATCH, *GRID)
    assert (out >= 0).all()


def test_csgd_head_returns_valid_parameters():
    torch.manual_seed(0)
    head = CsgdHead(DIM, PATCH, GRID)
    csgd = head(tokens())
    assert csgd.shape.shape == (BATCH, *GRID)
    assert (csgd.shape > 0).all()
    assert (csgd.scale > 0).all()
    assert (csgd.shift <= 0).all(), "a positive shift would remove the dry mass"


def test_linear_probe_has_no_blocks():
    """depth=0 asks what the frozen backbone already represents."""
    head = CsgdHead(DIM, PATCH, GRID, depth=0)
    assert len(head.head.blocks) == 0
    assert head(tokens()).shape.shape == (BATCH, *GRID)


def test_head_gradients_flow():
    torch.manual_seed(0)
    head = CsgdHead(DIM, PATCH, GRID, depth=1)
    csgd = head(tokens())
    target = torch.rand(BATCH, *GRID) * 3
    crps_sample(csgd.sample(8), target).mean().backward()
    for name, param in head.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient in {name}"
