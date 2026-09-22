"""Tests for ERA5 ingestion.

These run against a small fake store shaped like WeatherBench 2's ERA5 —
descending latitude, 13-style level axis, static fields — so CI never touches
the network and each test runs in well under a second.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from cirrus.data.ingest import (
    IngestSpec,
    check_available,
    check_layout,
    ingest,
    open_source,
)


@pytest.fixture
def fake_source(tmp_path: Path) -> str:
    """Build a tiny WB2-like store spanning a year boundary."""
    times = np.arange(
        np.datetime64("2009-12-30T00"),
        np.datetime64("2011-01-03T00"),
        np.timedelta64(6, "h"),
    ).astype("datetime64[ns]")
    # Mirror the real WeatherBench store's quirks: descending latitude AND
    # longitude stored before latitude. A fixture that is tidier than the real
    # source hides exactly the bugs it exists to catch.
    lat = np.linspace(87.1875, -87.1875, 8)
    lon = np.arange(16) * 22.5
    level = np.array([50, 250, 500, 850, 1000])
    rng = np.random.default_rng(0)

    def field(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape).astype("float64")

    nt, ny, nx, nz = len(times), len(lat), len(lon), len(level)
    ds = xr.Dataset(
        {
            "2m_temperature": (("time", "longitude", "latitude"), field(nt, nx, ny)),
            "total_precipitation_6hr": (
                ("time", "longitude", "latitude"),
                np.abs(field(nt, nx, ny)),
            ),
            "temperature": (
                ("time", "level", "longitude", "latitude"),
                field(nt, nz, nx, ny),
            ),
            "land_sea_mask": (("longitude", "latitude"), field(nx, ny)),
        },
        coords={"time": times, "latitude": lat, "longitude": lon, "level": level},
    )
    path = tmp_path / "source.zarr"
    ds.to_zarr(path, zarr_format=2)
    return str(path)


def make_spec(source: str, output: Path, start: str, end: str) -> IngestSpec:
    return IngestSpec(
        source=source,
        output=str(output),
        start=start,
        end=end,
        surface_variables=("2m_temperature", "total_precipitation_6hr"),
        upper_variables=("temperature",),
        levels=(250, 850),
        static_variables=("land_sea_mask",),
    )


def quiet(_: str) -> None:
    """Swallow progress logging in tests."""


def test_missing_variables_and_levels_all_reported(fake_source: str, tmp_path: Path):
    """Every problem is reported at once, not just the first."""
    spec = IngestSpec(
        source=fake_source,
        output=str(tmp_path / "out.zarr"),
        start="2010-01-01",
        end="2010-01-02",
        surface_variables=("2m_temperature", "not_a_variable"),
        upper_variables=("temperature",),
        levels=(300,),
    )
    with pytest.raises(ValueError) as err:
        check_available(open_source(fake_source), spec)
    assert "not_a_variable" in str(err.value)
    assert "300" in str(err.value)


def test_out_of_range_dates_rejected(fake_source: str, tmp_path: Path):
    spec = make_spec(fake_source, tmp_path / "o.zarr", "2008-01-01", "2010-01-02")
    with pytest.raises(ValueError, match="source covers"):
        check_available(open_source(fake_source), spec)


def test_malformed_date_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        IngestSpec(output=str(tmp_path / "o.zarr"), start="2010-1-1", end="2010-01-02")


def test_output_layout(fake_source: str, tmp_path: Path):
    """Right steps, right levels, ascending latitude, float32, statics untimed."""
    spec = make_spec(fake_source, tmp_path / "out.zarr", "2010-01-01", "2010-01-02")
    out = xr.open_zarr(ingest(spec, log=quiet), chunks=None)

    assert out.sizes["time"] == 8  # two days, 6-hourly
    assert list(out["level"].values) == [250, 850]
    assert np.all(np.diff(out["latitude"].values) > 0)
    assert out["temperature"].dims == ("time", "level", "latitude", "longitude")
    assert out["2m_temperature"].dims == ("time", "latitude", "longitude")
    assert out["temperature"].dtype == np.float32
    assert out["land_sea_mask"].dims == ("latitude", "longitude")
    check_layout(out)  # the consumer-side check must agree


def test_values_follow_their_coordinates(fake_source: str, tmp_path: Path):
    """Sorting latitude must move the data with it, not just the labels."""
    spec = make_spec(fake_source, tmp_path / "out.zarr", "2010-01-01", "2010-01-01")
    out = xr.open_zarr(ingest(spec, log=quiet), chunks=None)
    src = open_source(fake_source)

    point = {"time": "2010-01-01T06", "latitude": 87.1875, "longitude": 45.0}
    expected = float(src["2m_temperature"].sel(point))
    actual = float(out["2m_temperature"].sel(point))
    assert actual == pytest.approx(expected, rel=1e-6)


def test_spans_year_boundary(fake_source: str, tmp_path: Path):
    spec = make_spec(fake_source, tmp_path / "out.zarr", "2010-12-31", "2011-01-01")
    out = xr.open_zarr(ingest(spec, log=quiet), chunks=None)
    assert out.sizes["time"] == 8
    assert np.all(np.diff(out["time"].values) > np.timedelta64(0, "ns"))


def test_rerun_resumes_without_duplicates(fake_source: str, tmp_path: Path):
    """An interrupted-then-restarted ingest must not write any step twice."""
    output = tmp_path / "out.zarr"
    ingest(make_spec(fake_source, output, "2010-12-31", "2010-12-31"), log=quiet)
    ingest(make_spec(fake_source, output, "2010-12-31", "2011-01-01"), log=quiet)
    ingest(make_spec(fake_source, output, "2010-12-31", "2011-01-01"), log=quiet)

    out = xr.open_zarr(output, chunks=None)
    assert out.sizes["time"] == 8
    assert len(np.unique(out["time"].values)) == 8
