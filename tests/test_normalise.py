"""Tests for splits and normalisation.

Normalisation bugs are silent: a wrong mean or a leaked test year still
trains, and just produces worse results you cannot explain. So these tests
check properties, not just that the code runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from cirrus.data.ingest import IngestSpec
from cirrus.data.normalise import Normaliser, NormaliseSpec, compute_stats
from cirrus.data.splits import Period, Splits

TRAIN = Period("2000-01-01", "2001-12-31")
TEST_OFFSET = 1000.0


@pytest.fixture
def store(tmp_path: Path) -> tuple[str, IngestSpec]:
    """Build a small store in cirrus's canonical layout.

    Temperature is ~N(250, 10) in the training years but shifted by +1000 in
    2002, so any leakage of later years into the statistics is obvious.
    Precipitation is skewed and non-negative, with regridding-style negative
    residue sprinkled in.
    """
    times = np.arange(
        np.datetime64("2000-01-01T00"),
        np.datetime64("2003-01-01T00"),
        np.timedelta64(6, "h"),
    ).astype("datetime64[ns]")
    lat = np.linspace(-80.0, 80.0, 8)
    lon = np.arange(16) * 22.5
    level = np.array([500, 850])
    rng = np.random.default_rng(0)
    nt = len(times)

    t2m = 250.0 + 10.0 * rng.standard_normal((nt, 8, 16))
    t2m[times >= np.datetime64("2002-01-01")] += TEST_OFFSET

    tp = rng.exponential(0.002, (nt, 8, 16)) * (rng.random((nt, 8, 16)) < 0.3)
    tp[rng.random((nt, 8, 16)) < 0.01] = -2e-8

    ds = xr.Dataset(
        {
            "2m_temperature": (("time", "latitude", "longitude"), t2m),
            "total_precipitation_6hr": (("time", "latitude", "longitude"), tp),
            "temperature": (
                ("time", "level", "latitude", "longitude"),
                rng.standard_normal((nt, 2, 8, 16)),
            ),
            "land_sea_mask": (("latitude", "longitude"), rng.random((8, 16))),
        },
        coords={"time": times, "latitude": lat, "longitude": lon, "level": level},
    ).astype("float32")
    path = tmp_path / "store.zarr"
    ds.to_zarr(path, zarr_format=2)

    spec = IngestSpec(
        output=str(path),
        start="2000-01-01",
        end="2002-12-31",
        surface_variables=("2m_temperature", "total_precipitation_6hr"),
        upper_variables=("temperature",),
        levels=(500, 850),
        static_variables=("land_sea_mask",),
    )
    return str(path), spec


def quiet(_: str) -> None:
    """Swallow progress logging in tests."""


def stack(ds: xr.Dataset, spec: IngestSpec) -> np.ndarray:
    """Build a (time, channel, lat, lon) array in canonical channel order."""
    arrays = [ds[v].values for v in spec.surface_variables]
    arrays += [
        ds[v].sel(level=lev).values for v in spec.upper_variables for lev in spec.levels
    ]
    return np.stack(arrays, axis=1)


@pytest.fixture
def normaliser(store: tuple[str, IngestSpec]) -> Normaliser:
    path, spec = store
    return compute_stats(path, spec, TRAIN, NormaliseSpec(), log=quiet)


def test_channel_order_is_the_contract(normaliser: Normaliser):
    assert normaliser.channels == (
        "2m_temperature",
        "total_precipitation_6hr",
        "temperature_500",
        "temperature_850",
        "land_sea_mask",
    )
    assert normaliser.log_channels == ("total_precipitation_6hr",)


def test_training_data_is_standardised(
    store: tuple[str, IngestSpec], normaliser: Normaliser
):
    """On the training period, every channel has weighted mean 0 and std 1."""
    path, spec = store
    ds = xr.open_zarr(path, chunks=None).sel(time=slice(TRAIN.start, TRAIN.end))
    z = normaliser.subset(spec.time_channels).normalise(stack(ds, spec))

    w = np.cos(np.deg2rad(ds["latitude"].values))[:, None] * np.ones(16)
    w = np.broadcast_to(w, z.shape[:1] + w.shape)
    for c in range(z.shape[1]):
        channel = z[:, c].astype(np.float64)
        mean = (channel * w).sum() / w.sum()
        std = np.sqrt(((channel - mean) ** 2 * w).sum() / w.sum())
        assert mean == pytest.approx(0.0, abs=1e-4)
        assert std == pytest.approx(1.0, abs=1e-4)


def test_test_period_does_not_leak(normaliser: Normaliser):
    """2002 is shifted by +1000; a training-only mean stays near 250."""
    mean = normaliser.mean[normaliser.channels.index("2m_temperature")]
    assert mean == pytest.approx(250.0, abs=1.0)


def test_round_trip_is_exact(store: tuple[str, IngestSpec], normaliser: Normaliser):
    path, spec = store
    ds = xr.open_zarr(path, chunks=None).sel(time=slice("2000-06-01", "2000-06-02"))
    x = stack(ds, spec)
    x[:, 1] = np.clip(x[:, 1], 0.0, None)  # round trip holds for valid precip
    time_norm = normaliser.subset(spec.time_channels)
    back = time_norm.denormalise(time_norm.normalise(x))
    np.testing.assert_allclose(back, x, rtol=1e-4, atol=1e-7)


def test_negative_precipitation_residue_is_clipped(normaliser: Normaliser):
    precip = normaliser.subset(["total_precipitation_6hr"])
    tiny_negative = np.full((1, 2, 2), -2e-8, dtype=np.float32)
    zero = np.zeros((1, 2, 2), dtype=np.float32)
    np.testing.assert_array_equal(
        precip.normalise(tiny_negative), precip.normalise(zero)
    )


def test_denormalised_precipitation_never_negative(normaliser: Normaliser):
    """A model output far below the floor must not become negative rain."""
    precip = normaliser.subset(["total_precipitation_6hr"])
    very_low = np.full((1, 2, 2), -50.0, dtype=np.float32)
    assert (precip.denormalise(very_low) >= 0).all()


def test_log_transform_compresses_the_tail(normaliser: Normaliser):
    """40 mm must not sit forty times further out than 1 mm."""
    precip = normaliser.subset(["total_precipitation_6hr"])
    values = np.array([0.0, 0.001, 0.040], dtype=np.float32).reshape(3, 1, 1, 1)
    z = precip.normalise(values).ravel()
    gap_small, gap_large = z[1] - z[0], z[2] - z[1]
    assert gap_large < 5 * gap_small


def test_save_load_round_trip(normaliser: Normaliser, tmp_path: Path):
    loaded = Normaliser.load(normaliser.save(tmp_path / "stats.json"))
    assert loaded.channels == normaliser.channels
    assert loaded.log_channels == normaliser.log_channels
    np.testing.assert_array_equal(loaded.mean, normaliser.mean)
    np.testing.assert_array_equal(loaded.std, normaliser.std)


def test_wrong_channel_count_rejected(normaliser: Normaliser):
    with pytest.raises(ValueError, match="channels on axis -3"):
        normaliser.normalise(np.zeros((3, 8, 16), dtype=np.float32))


def test_splits_must_not_overlap():
    with pytest.raises(ValueError, match="before validation"):
        Splits(
            train=Period("2000-01-01", "2010-12-31"),
            val=Period("2010-06-01", "2011-12-31"),
            test=Period("2012-01-01", "2013-12-31"),
        )


def test_project_splits_are_valid():
    """The committed split config must itself pass validation."""
    splits = Splits.from_yaml("configs/data/splits.yaml")
    assert splits.test.start == "2017-01-01"


def test_non_canonical_store_is_rejected(tmp_path: Path):
    """A (time, lon, lat) store must fail loudly, not broadcast silently."""
    ds = xr.Dataset(
        {"2m_temperature": (("time", "longitude", "latitude"), np.zeros((4, 16, 8)))},
        coords={
            "time": np.arange(4).astype("datetime64[D]").astype("datetime64[ns]"),
            "longitude": np.arange(16) * 22.5,
            "latitude": np.linspace(-80.0, 80.0, 8),
        },
    )
    path = tmp_path / "bad.zarr"
    ds.to_zarr(path, zarr_format=2)
    spec = IngestSpec(
        output=str(path),
        start="1970-01-01",
        end="1970-01-04",
        surface_variables=("2m_temperature",),
    )
    train = Period("1970-01-01", "1970-01-04")
    with pytest.raises(ValueError, match="canonical layout"):
        compute_stats(path, spec, train, NormaliseSpec(), log=quiet)
