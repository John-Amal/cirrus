"""Tests for the extreme value machinery.

These are checkable against ground truth in a way that model results never
are: data is generated from a known generalised Pareto, so the right answer
for every fit and every return level is known in advance.

The block bootstrap test is the one worth reading. With serially correlated
data, an ordinary bootstrap treats neighbouring values as independent and
reports intervals that are too narrow. The block version must be wider --
otherwise the uncertainty on every observed return level in this project is
understated.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import genpareto

from cirrus.eval.gpd import (
    MIN_EXCEEDANCES,
    GpdFit,
    bootstrap_return_levels,
    fit_field,
    fit_gpd,
    return_level_field,
)


def synthetic(shape: float, scale: float, threshold: float, n: int, seed: int = 0):
    """Values whose exceedances of ``threshold`` are exactly GPD(shape, scale)."""
    rng = np.random.default_rng(seed)
    below = rng.uniform(0.0, threshold, size=n * 9)
    above = threshold + genpareto.rvs(
        shape, scale=scale, size=n, random_state=np.random.default_rng(seed + 1)
    )
    values = np.concatenate([below, above])
    rng.shuffle(values)
    return values


def test_recovers_known_parameters():
    values = synthetic(shape=0.15, scale=2.0, threshold=10.0, n=5_000)
    fit = fit_gpd(values, threshold=10.0)
    assert fit.shape == pytest.approx(0.15, abs=0.05)
    assert fit.scale == pytest.approx(2.0, abs=0.2)
    assert fit.exceedance_rate == pytest.approx(0.1, abs=0.01)


def test_fixed_shape_is_respected():
    values = synthetic(shape=0.15, scale=2.0, threshold=10.0, n=2_000)
    fit = fit_gpd(values, threshold=10.0, fixed_shape=0.0)
    assert fit.shape == pytest.approx(0.0, abs=1e-9)
    assert fit.scale > 0


def test_too_few_exceedances_gives_nan_not_a_number():
    """A confident-looking estimate from 5 points would be worse than none."""
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.uniform(0, 10, 1_000), np.array([11.0, 12.0, 13.0])])
    fit = fit_gpd(values, threshold=10.0)
    assert not fit.is_valid
    assert np.isnan(fit.return_level(20))
    assert fit.n_exceedances < MIN_EXCEEDANCES


def test_return_level_matches_the_empirical_quantile():
    """With a large sample the fitted 1-year level must match what happened."""
    shape, scale, threshold = 0.1, 3.0, 20.0
    values = synthetic(shape, scale, threshold, n=40_000, seed=3)
    fit = fit_gpd(values, threshold)

    blocks = 1460  # one year of 6-hourly steps
    expected_exceedances = blocks * fit.exceedance_rate
    empirical = float(np.quantile(values, 1.0 - 1.0 / blocks))
    fitted = fit.return_level(1.0)

    assert expected_exceedances > 1
    assert fitted == pytest.approx(empirical, rel=0.15)


def test_exponential_limit_is_handled():
    """Xi = 0 needs the log form, not a division by zero."""
    fit = GpdFit(
        shape=0.0, scale=2.0, threshold=10.0, exceedance_rate=0.02, n_exceedances=500
    )
    expected = 10.0 + 2.0 * np.log(20 * 1460 * 0.02)
    assert fit.return_level(20) == pytest.approx(expected)


def test_heavier_shape_gives_higher_return_levels():
    common = {
        "scale": 2.0,
        "threshold": 10.0,
        "exceedance_rate": 0.02,
        "n_exceedances": 500,
    }
    light = GpdFit(shape=-0.1, **common).return_level(20)
    medium = GpdFit(shape=0.0, **common).return_level(20)
    heavy = GpdFit(shape=0.2, **common).return_level(20)
    assert light < medium < heavy


def test_short_return_period_falls_back_to_the_threshold():
    """If the period is shorter than the gap between exceedances, u is it."""
    fit = GpdFit(
        shape=0.1, scale=2.0, threshold=10.0, exceedance_rate=1e-5, n_exceedances=100
    )
    assert fit.return_level(0.01) == 10.0


def test_block_bootstrap_is_wider_on_correlated_data():
    """The point of blocks: clustered exceedances are not independent."""
    rng = np.random.default_rng(0)
    base = synthetic(0.1, 2.0, 10.0, n=3_000, seed=5)
    # Impose serial dependence: repeat runs of values, as storms do.
    correlated = np.repeat(base[::4], 4)[: len(base)]
    correlated = correlated + rng.normal(0, 0.01, correlated.shape)

    iid_low, _, iid_high = bootstrap_return_levels(
        correlated, 10.0, years=5, n_resamples=120, block_size=None, seed=1
    )
    block_low, _, block_high = bootstrap_return_levels(
        correlated, 10.0, years=5, n_resamples=120, block_size=120, seed=1
    )
    assert (block_high - block_low) > (iid_high - iid_low)


def test_bootstrap_interval_brackets_the_point_estimate():
    values = synthetic(0.1, 2.0, 10.0, n=4_000, seed=7)
    low, point, high = bootstrap_return_levels(values, 10.0, years=5, n_resamples=120)
    assert low < point < high


def test_bootstrap_covers_the_truth():
    """A 90% interval should contain the true value most of the time."""
    shape, scale, threshold = 0.1, 2.0, 10.0
    truth = GpdFit(shape, scale, threshold, 0.1, 10_000).return_level(5)
    covered = 0
    for seed in range(12):
        values = synthetic(shape, scale, threshold, n=3_000, seed=100 + seed)
        low, _, high = bootstrap_return_levels(
            values, threshold, years=5, n_resamples=80, seed=seed
        )
        covered += int(low <= truth <= high)
    assert covered >= 8  # allowing for a small number of trials


def test_fit_field_shapes_and_missing_cells():
    rng = np.random.default_rng(0)
    values = np.stack(
        [
            synthetic(0.1, 2.0, 10.0, n=400, seed=1).reshape(-1, 1, 1)[:4000, 0, 0]
            for _ in range(1)
        ]
    ).reshape(-1, 1, 1)
    values = np.tile(values, (1, 2, 3))
    values[:, 1, 2] = rng.uniform(0, 5, values.shape[0])  # never exceeds
    thresholds = np.full((2, 3), 10.0)

    shapes, scales, counts = fit_field(values, thresholds)
    assert shapes.shape == (2, 3)
    assert np.isnan(shapes[1, 2]), "a cell with no exceedances must be NaN"
    assert np.isfinite(shapes[0, 0])
    assert counts[0, 0] > MIN_EXCEEDANCES


def test_fit_field_rejects_mismatched_thresholds():
    with pytest.raises(ValueError, match="do not match"):
        fit_field(np.zeros((10, 2, 3)), np.zeros((4, 4)))


def test_return_level_field_matches_the_scalar_version():
    shapes = np.array([[0.1, 0.0]])
    scales = np.array([[2.0, 2.0]])
    thresholds = np.array([[10.0, 10.0]])
    rates = np.array([[0.02, 0.02]])

    field = return_level_field(shapes, scales, thresholds, rates, years=20)
    for column in (0, 1):
        scalar = GpdFit(
            float(shapes[0, column]),
            float(scales[0, column]),
            float(thresholds[0, column]),
            float(rates[0, column]),
            n_exceedances=500,
        ).return_level(20)
        assert field[0, column] == pytest.approx(scalar, rel=1e-6)
