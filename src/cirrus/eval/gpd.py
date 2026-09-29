"""Generalised Pareto fits and return levels.

The amplitude ratios in Phase 3 say whether a model can produce intense
values. Return levels say something a risk calculation can use: the level
exceeded once in N years.

Estimated by peaks over threshold. Above a high enough threshold, exceedances
of almost any distribution converge to a generalised Pareto with shape ``xi``
and scale ``sigma``, and the return level follows in closed form. The shape
governs how the tail decays: ``xi > 0`` is heavy (unbounded), ``xi = 0`` is
exponential, ``xi < 0`` has a finite upper limit.

Two choices here are the awkward ones, so they are made explicit rather than
buried.

**The shape parameter can be fixed.** Per-cell estimates of ``xi`` are noisy
-- a few hundred exceedances gives a standard error of order 0.1 -- and that
noise propagates hard into a 20-year return level. Fixing ``xi`` from a
longer record cuts the variance, and because the *same* value is applied to
both model and observations, an error in it largely cancels in their ratio.
It must not be the only analysis, though: fixing ``xi`` hands the model the
observed tail shape, which would hide a model whose deficiency is in the
shape rather than the scale. :func:`fit_gpd` supports both, and the caller is
expected to report both.

**Intervals come from a block bootstrap on observations.** Six-hourly
precipitation exceedances cluster -- one storm contributes several -- which
violates the independence a GPD fit assumes and makes an ordinary bootstrap
understate uncertainty. Resampling contiguous blocks preserves that
clustering. Model draws need no such treatment: a sample from a per-cell
predictive distribution has no temporal coherence, so its exceedances really
are independent. The asymmetry is real and worth stating in any write-up
rather than papering over.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import genpareto

MIN_EXCEEDANCES = 30  # below this a fit is noise dressed as a number
BLOCKS_PER_YEAR = 1460  # 6-hourly steps


@dataclass(frozen=True)
class GpdFit:
    """A peaks-over-threshold fit for one location."""

    shape: float
    scale: float
    threshold: float
    exceedance_rate: float
    n_exceedances: int

    @property
    def is_valid(self) -> bool:
        """Whether the fit rests on enough data to mean anything."""
        return self.n_exceedances >= MIN_EXCEEDANCES and np.isfinite(self.scale)

    def return_level(
        self, years: float, blocks_per_year: int = BLOCKS_PER_YEAR
    ) -> float:
        """Level exceeded once per ``years``, in the units of the data.

        The standard POT expression: with ``m`` blocks in the return period
        and ``zeta`` the probability of exceeding the threshold,

            x = u + (sigma / xi) * ((m * zeta)^xi - 1)

        and its ``xi -> 0`` limit ``x = u + sigma * log(m * zeta)``.
        """
        if not self.is_valid:
            return float("nan")
        expected = years * blocks_per_year * self.exceedance_rate
        if expected <= 1.0:
            # The return period is shorter than the average gap between
            # exceedances, so the threshold itself is the answer.
            return self.threshold
        if abs(self.shape) < 1e-6:
            return self.threshold + self.scale * float(np.log(expected))
        return self.threshold + (self.scale / self.shape) * (
            float(expected**self.shape) - 1.0
        )


def fit_gpd(
    values: np.ndarray,
    threshold: float,
    fixed_shape: float | None = None,
) -> GpdFit:
    """Fit a GPD to exceedances of ``threshold``.

    Args:
        values: All values for one location, not just the exceedances.
        threshold: The threshold, in the same units.
        fixed_shape: Hold ``xi`` at this value and fit only the scale. Pass
            None to estimate both.

    Returns:
        The fit. Check :attr:`GpdFit.is_valid` before using it: too few
        exceedances yields a fit that is numerically fine and statistically
        meaningless.

    """
    finite = values[np.isfinite(values)]
    exceedances = finite[finite > threshold] - threshold
    rate = len(exceedances) / max(len(finite), 1)

    if len(exceedances) < MIN_EXCEEDANCES:
        return GpdFit(float("nan"), float("nan"), threshold, rate, len(exceedances))

    if fixed_shape is None:
        shape, _, scale = genpareto.fit(exceedances, floc=0.0)
    else:
        shape, _, scale = genpareto.fit(exceedances, f0=fixed_shape, floc=0.0)
    return GpdFit(float(shape), float(scale), threshold, rate, len(exceedances))


def bootstrap_return_levels(
    values: np.ndarray,
    threshold: float,
    years: float,
    n_resamples: int = 200,
    block_size: int | None = None,
    fixed_shape: float | None = None,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return level with a bootstrap interval, as ``(low, point, high)``.

    Args:
        values: All values for one location.
        threshold: Exceedance threshold, same units.
        years: Return period.
        n_resamples: Bootstrap replicates.
        block_size: Resample contiguous blocks of this length, preserving
            serial dependence. Use it for observations; leave it None for
            model draws, which are independent by construction.
        fixed_shape: Hold ``xi`` fixed, as in :func:`fit_gpd`.
        seed: Seed for the resampling.

    The interval is the 5th and 95th percentiles of the resampled estimates.

    """
    point = fit_gpd(values, threshold, fixed_shape).return_level(years)
    rng = np.random.default_rng(seed)
    n = len(values)
    estimates = np.empty(n_resamples)

    for index in range(n_resamples):
        if block_size and block_size < n:
            n_blocks = int(np.ceil(n / block_size))
            starts = rng.integers(0, n - block_size, size=n_blocks)
            resample = np.concatenate([values[s : s + block_size] for s in starts])[:n]
        else:
            resample = values[rng.integers(0, n, size=n)]
        estimates[index] = fit_gpd(resample, threshold, fixed_shape).return_level(years)

    usable = estimates[np.isfinite(estimates)]
    if usable.size < n_resamples // 4:
        return float("nan"), point, float("nan")
    return float(np.percentile(usable, 5)), point, float(np.percentile(usable, 95))


def fit_field(
    values: np.ndarray,
    thresholds: np.ndarray,
    fixed_shapes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit every grid cell of a ``(time, lat, lon)`` array.

    Returns ``(shape, scale, n_exceedances)``, each ``(lat, lon)``. Cells with
    too few exceedances come back as NaN rather than as a confident-looking
    number.
    """
    if values.ndim != 3:
        raise ValueError(f"expected (time, lat, lon), got {values.shape}")
    if thresholds.shape != values.shape[1:]:
        raise ValueError(
            f"thresholds {thresholds.shape} do not match grid {values.shape[1:]}"
        )

    n_lat, n_lon = thresholds.shape
    shapes = np.full((n_lat, n_lon), np.nan)
    scales = np.full((n_lat, n_lon), np.nan)
    counts = np.zeros((n_lat, n_lon), dtype=int)

    for i in range(n_lat):
        for j in range(n_lon):
            fixed = None if fixed_shapes is None else float(fixed_shapes[i, j])
            if fixed is not None and not np.isfinite(fixed):
                fixed = None
            fit = fit_gpd(values[:, i, j], float(thresholds[i, j]), fixed)
            counts[i, j] = fit.n_exceedances
            if fit.is_valid:
                shapes[i, j] = fit.shape
                scales[i, j] = fit.scale
    return shapes, scales, counts


def return_level_field(
    shapes: np.ndarray,
    scales: np.ndarray,
    thresholds: np.ndarray,
    rates: np.ndarray,
    years: float,
    blocks_per_year: int = BLOCKS_PER_YEAR,
) -> np.ndarray:
    """Return levels for a whole grid, vectorised."""
    expected = years * blocks_per_year * rates
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        heavy = thresholds + (scales / shapes) * (expected**shapes - 1.0)
        light = thresholds + scales * np.log(np.maximum(expected, 1e-12))
        levels = np.where(np.abs(shapes) < 1e-6, light, heavy)
        levels = np.where(expected <= 1.0, thresholds, levels)
    return np.where(np.isfinite(scales), levels, np.nan)
