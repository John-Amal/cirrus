"""Tests for the reference forecasts.

The important one is ``test_sampling_reproduces_the_distribution``. The
quantile levels are deliberately uneven -- dense in the tail, so the far end
is resolved -- and that makes the obvious sampling implementation wrong:
drawing a uniform *index* oversamples the tail by a factor of several,
because 60 of the 460 levels sit above the 99th percentile. The first version
of this code did exactly that and inflated the mean 3.6-fold. Sampling has to
happen in probability space.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from cirrus.data.ingest import IngestSpec
from cirrus.data.splits import Period
from cirrus.eval.baselines import (
    QUANTILE_LEVELS,
    Climatology,
    ClimatologyBaseline,
    PersistenceBaseline,
    build_climatology,
    months_from_nanoseconds,
)

GRID = (4, 8)
TRAIN = Period("2000-01-01", "2003-12-31")
JANUARY, JUNE = 0, 5  # zero-based, matching how the climatology is indexed


@pytest.fixture
def store(tmp_path: Path) -> tuple[str, IngestSpec]:
    """Four years of intermittent precipitation with a seasonal cycle."""
    times = np.arange(
        np.datetime64("2000-01-01T00"),
        np.datetime64("2004-01-01T00"),
        np.timedelta64(6, "h"),
    ).astype("datetime64[ns]")
    rng = np.random.default_rng(0)
    n = len(times)

    # Zero-based: numpy's month arithmetic makes January 0, so June is 5.
    # The climatology stores months at dt.month - 1, which agrees.
    month_index = times.astype("datetime64[M]").astype(int) % 12
    wet_season = 0.15 + 0.25 * (month_index == JUNE)
    wet = rng.random((n, *GRID)) < wet_season[:, None, None]
    values = (rng.exponential(0.003, (n, *GRID)) * wet).astype(np.float32)

    ds = xr.Dataset(
        {"total_precipitation_6hr": (("time", "latitude", "longitude"), values)},
        coords={
            "time": times,
            "latitude": np.linspace(-60.0, 60.0, GRID[0]),
            "longitude": np.arange(GRID[1]) * 45.0,
        },
    )
    path = tmp_path / "store.zarr"
    ds.to_zarr(path, zarr_format=2)
    spec = IngestSpec(
        output=str(path),
        start="2000-01-01",
        end="2003-12-31",
        surface_variables=("total_precipitation_6hr",),
    )
    return str(path), spec


def test_climatology_shape_and_metadata(store: tuple[str, IngestSpec]):
    path, spec = store
    climatology = build_climatology(path, spec, TRAIN)
    assert climatology.quantiles.shape == (12, GRID[0] * GRID[1], len(QUANTILE_LEVELS))
    assert climatology.meta["units"] == "mm"
    assert climatology.meta["grid"] == list(GRID)


def test_quantiles_increase_with_level(store: tuple[str, IngestSpec]):
    path, spec = store
    climatology = build_climatology(path, spec, TRAIN)
    differences = np.diff(climatology.quantiles, axis=-1)
    assert (differences >= -1e-6).all(), "quantiles must be non-decreasing"


def test_seasonal_cycle_is_captured(store: tuple[str, IngestSpec]):
    """June rains more often in the fixture, so its 80th percentile is higher.

    Compared at a mid-upper quantile on purpose. The fixture varies how
    *often* it rains, not how hard, so the extreme quantiles carry almost no
    seasonal signal and are dominated by sampling noise in a few values.
    """
    path, spec = store
    climatology = build_climatology(path, spec, TRAIN)
    level = int(np.searchsorted(QUANTILE_LEVELS, 0.8))

    june = climatology.quantiles[JUNE, :, level].mean()
    january = climatology.quantiles[JANUARY, :, level].mean()
    assert june > january


def test_sampling_reproduces_the_distribution():
    """Uneven levels make uniform-index sampling wrong; this pins the fix."""
    rng = np.random.default_rng(0)
    truth = rng.exponential(2.0, 100_000) * (rng.random(100_000) < 0.3)
    table = np.quantile(truth, QUANTILE_LEVELS).astype(np.float32)

    climatology = Climatology(
        quantiles=np.tile(table, (12, 1, 1)),
        levels=QUANTILE_LEVELS.astype(np.float32),
        meta={},
    )
    baseline = ClimatologyBaseline(climatology, torch.device("cpu"))
    times = torch.full((64,), int(np.datetime64("2001-03-15T00", "ns").astype("int64")))
    drawn = baseline.samples(times, (1, 1), draws=200).numpy().ravel()

    assert drawn.mean() == pytest.approx(truth.mean(), rel=0.1)
    assert (drawn == 0).mean() == pytest.approx((truth == 0).mean(), abs=0.03)
    for q in (0.9, 0.99):
        assert np.quantile(drawn, q) == pytest.approx(np.quantile(truth, q), rel=0.12)
    # Looser in the far tail on purpose: 12,800 draws put only ~13 values
    # above the 99.9th percentile, so a tight bound here would fail
    # intermittently for sampling reasons rather than for code reasons.
    assert np.quantile(drawn, 0.999) == pytest.approx(
        np.quantile(truth, 0.999), rel=0.3
    )


def test_uniform_index_sampling_would_be_wrong():
    """Guard the guard: the naive implementation must visibly fail.

    If this ever stops failing, the levels have become evenly spaced and the
    tail resolution that motivated them has been lost.
    """
    rng = np.random.default_rng(0)
    truth = rng.exponential(2.0, 100_000) * (rng.random(100_000) < 0.3)
    table = np.quantile(truth, QUANTILE_LEVELS)
    naive = table[rng.integers(0, len(QUANTILE_LEVELS), 100_000)]
    assert naive.mean() > 2 * truth.mean()


def test_sampling_shape_and_non_negativity():
    climatology = Climatology(
        quantiles=np.abs(
            np.random.default_rng(0).standard_normal((12, 32, 460))
        ).astype(np.float32),
        levels=QUANTILE_LEVELS.astype(np.float32),
        meta={},
    )
    baseline = ClimatologyBaseline(climatology, torch.device("cpu"))
    times = torch.full((3,), int(np.datetime64("2001-07-01T00", "ns").astype("int64")))
    drawn = baseline.samples(times, (4, 8), draws=5)
    assert drawn.shape == (3, 4, 8, 5)
    assert (drawn >= 0).all()


def test_months_are_extracted_correctly():
    dates = ["2001-01-15T06", "2001-06-30T18", "2001-12-01T00", "2004-02-29T12"]
    times = torch.tensor(
        [int(np.datetime64(d, "ns").astype("int64")) for d in dates], dtype=torch.long
    )
    assert months_from_nanoseconds(times).tolist() == [0, 5, 11, 1]


def test_persistence_repeats_its_value():
    field = torch.rand(2, 4, 8) * 5
    samples = PersistenceBaseline(channel=0).samples(field, draws=7)
    assert samples.shape == (2, 4, 8, 7)
    for draw in range(7):
        torch.testing.assert_close(samples[..., draw], field)


def test_save_load_round_trip(store: tuple[str, IngestSpec], tmp_path: Path):
    path, spec = store
    climatology = build_climatology(path, spec, TRAIN)
    loaded = Climatology.load(climatology.save(tmp_path / "clim.npz"))
    np.testing.assert_allclose(loaded.quantiles, climatology.quantiles)
    assert loaded.meta == climatology.meta
