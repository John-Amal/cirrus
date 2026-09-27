"""Tests for exceedance thresholds.

The property that matters is the achieved exceedance rate: a threshold set to
give 2% of steps must actually give about 2%, in wet cells and dry ones
alike. That is what makes per-cell tail statistics comparable, and it is the
whole reason for choosing a rate-based definition over a fixed quantile.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from cirrus.data.ingest import IngestSpec
from cirrus.data.splits import Period
from cirrus.data.thresholds import Thresholds, ThresholdSpec, compute_thresholds

TRAIN = Period("2000-01-01", "2001-12-31")
LATS, LONS = 4, 8


@pytest.fixture
def store(tmp_path: Path) -> tuple[str, IngestSpec]:
    """Build a store with cells ranging from very wet to bone dry.

    Row 0 rains often, row 3 almost never, so the floor and the quantile
    branches are both exercised. The test period carries absurd values, to
    catch any leakage into a training-only statistic.
    """
    times = np.arange(
        np.datetime64("2000-01-01T00"),
        np.datetime64("2003-01-01T00"),
        np.timedelta64(6, "h"),
    ).astype("datetime64[ns]")
    rng = np.random.default_rng(0)
    n_steps = len(times)

    wet_fraction = np.array([0.5, 0.2, 0.05, 0.001])[:, None]
    rain = rng.exponential(0.004, (n_steps, LATS, LONS))
    is_wet = rng.random((n_steps, LATS, LONS)) < wet_fraction[None]
    values = np.where(is_wet, rain, 0.0).astype(np.float32)
    values[rng.random((n_steps, LATS, LONS)) < 0.01] = -2e-8  # regridding residue
    values[times >= np.datetime64("2002-01-01")] = 1.0  # 1000 mm, test period only

    ds = xr.Dataset(
        {"total_precipitation_6hr": (("time", "latitude", "longitude"), values)},
        coords={
            "time": times,
            "latitude": np.linspace(-60.0, 60.0, LATS),
            "longitude": np.arange(LONS) * 45.0,
        },
    )
    path = tmp_path / "store.zarr"
    ds.to_zarr(path, zarr_format=2)
    spec = IngestSpec(
        output=str(path),
        start="2000-01-01",
        end="2002-12-31",
        surface_variables=("total_precipitation_6hr",),
    )
    return str(path), spec


def quiet(_: str) -> None:
    """Swallow progress logging."""


def test_achieved_rate_matches_the_target(store: tuple[str, IngestSpec]):
    """The defining property: every cell gets ~the requested exceedance rate."""
    path, spec = store
    tspec = ThresholdSpec(exceedance_rate=0.05, floor=0.0)
    thresholds = compute_thresholds(path, spec, TRAIN, tspec, quiet)

    data = xr.open_zarr(path, chunks=None)["total_precipitation_6hr"]
    values = (
        np.clip(data.sel(time=slice(TRAIN.start, TRAIN.end)).values, 0, None) * 1000
    )
    rates = (values > thresholds.values[None]).mean(axis=0)

    wet_cells = rates[:2]  # rows dry enough to be all zeros cannot hit the rate
    np.testing.assert_allclose(wet_cells, 0.05, atol=0.01)


def test_floor_is_applied_to_dry_cells(store: tuple[str, IngestSpec]):
    path, spec = store
    tspec = ThresholdSpec(exceedance_rate=0.02, floor=0.5)
    thresholds = compute_thresholds(path, spec, TRAIN, tspec, quiet)

    assert (thresholds.values >= 0.5).all()
    assert thresholds.floored_fraction > 0  # the driest row must be floored
    assert thresholds.values[0].mean() > thresholds.values[3].mean()


def test_thresholds_use_training_years_only(store: tuple[str, IngestSpec]):
    """The test period is 1000 mm everywhere; thresholds must not notice."""
    path, spec = store
    thresholds = compute_thresholds(path, spec, TRAIN, ThresholdSpec(), quiet)
    assert thresholds.values.max() < 100.0


def test_negative_residue_does_not_shift_the_distribution(
    store: tuple[str, IngestSpec],
):
    """Regridding leaves tiny negatives; they are clipped, not counted as dry."""
    path, spec = store
    thresholds = compute_thresholds(path, spec, TRAIN, ThresholdSpec(floor=0.0), quiet)
    assert (thresholds.values >= 0).all()


def test_shape_and_metadata(store: tuple[str, IngestSpec]):
    path, spec = store
    thresholds = compute_thresholds(path, spec, TRAIN, ThresholdSpec(), quiet)
    assert thresholds.values.shape == (LATS, LONS)
    assert thresholds.meta["units"] == "mm"
    assert thresholds.meta["period"] == [TRAIN.start, TRAIN.end]
    assert thresholds.meta["events_per_cell_median"] > 0


def test_save_load_round_trip(store: tuple[str, IngestSpec], tmp_path: Path):
    path, spec = store
    thresholds = compute_thresholds(path, spec, TRAIN, ThresholdSpec(), quiet)
    loaded = Thresholds.load(thresholds.save(tmp_path / "t.json"))
    np.testing.assert_allclose(loaded.values, thresholds.values)
    assert loaded.meta == thresholds.meta


def test_bad_rates_are_rejected():
    with pytest.raises(ValueError, match="exceedance_rate"):
        ThresholdSpec(exceedance_rate=0.0)
    with pytest.raises(ValueError, match="exceedance_rate"):
        ThresholdSpec(exceedance_rate=0.8)
    with pytest.raises(ValueError, match="floor"):
        ThresholdSpec(floor=-1.0)


def test_missing_variable_is_reported(store: tuple[str, IngestSpec]):
    path, spec = store
    with pytest.raises(ValueError, match="not in"):
        compute_thresholds(path, spec, TRAIN, ThresholdSpec(variable="snow"), quiet)


def test_empty_period_is_reported(store: tuple[str, IngestSpec]):
    path, spec = store
    with pytest.raises(ValueError, match="no data"):
        compute_thresholds(
            path, spec, Period("1990-01-01", "1990-12-31"), ThresholdSpec(), quiet
        )
