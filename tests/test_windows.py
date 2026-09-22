"""Tests for sample windowing.

The failure modes here are silent. An off-by-one in the target index trains a
model to predict the wrong lead time and still converges; a window crossing
the train/test boundary leaks and merely makes your scores too good. So these
tests pin down indices and boundaries explicitly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from cirrus.data.ingest import IngestSpec
from cirrus.data.normalise import Normaliser, NormaliseSpec, compute_stats
from cirrus.data.splits import Period
from cirrus.data.windows import (
    FORCING_CHANNELS,
    WindowSource,
    WindowSpec,
    time_encodings,
    valid_starts,
)

TRAIN = Period("2000-01-01", "2000-12-31")


def build_store(path: Path, drop: slice | None = None) -> tuple[str, IngestSpec]:
    """Build a canonical-layout store; ``drop`` removes steps to make a gap."""
    times = np.arange(
        np.datetime64("2000-01-01T00"),
        np.datetime64("2001-06-01T00"),
        np.timedelta64(6, "h"),
    ).astype("datetime64[ns]")
    if drop is not None:
        times = np.delete(times, drop)
    lat = np.linspace(-80.0, 80.0, 8)
    lon = np.arange(16) * 22.5
    level = np.array([500, 850])
    rng = np.random.default_rng(0)
    nt = len(times)

    ds = xr.Dataset(
        {
            "2m_temperature": (
                ("time", "latitude", "longitude"),
                250 + 10 * rng.standard_normal((nt, 8, 16)),
            ),
            "total_precipitation_6hr": (
                ("time", "latitude", "longitude"),
                rng.exponential(0.002, (nt, 8, 16)),
            ),
            "temperature": (
                ("time", "level", "latitude", "longitude"),
                rng.standard_normal((nt, 2, 8, 16)),
            ),
            "land_sea_mask": (("latitude", "longitude"), rng.random((8, 16))),
        },
        coords={"time": times, "latitude": lat, "longitude": lon, "level": level},
    ).astype("float32")
    ds.to_zarr(path, zarr_format=2)

    spec = IngestSpec(
        output=str(path),
        start="2000-01-01",
        end="2001-05-31",
        surface_variables=("2m_temperature", "total_precipitation_6hr"),
        upper_variables=("temperature",),
        levels=(500, 850),
        static_variables=("land_sea_mask",),
    )
    return str(path), spec


def quiet(_: str) -> None:
    """Swallow progress logging in tests."""


@pytest.fixture
def source(tmp_path: Path) -> WindowSource:
    path, spec = build_store(tmp_path / "store.zarr")
    norm = compute_stats(path, spec, TRAIN, NormaliseSpec(), log=quiet)
    return WindowSource(path, spec, TRAIN, WindowSpec(), norm)


def test_valid_starts_counts_windows():
    times = np.arange(
        np.datetime64("2000-01-01T00"),
        np.datetime64("2000-01-03T00"),
        np.timedelta64(6, "h"),
    )
    assert len(valid_starts(times, span=3)) == len(times) - 2
    assert len(valid_starts(times, span=len(times) + 1)) == 0


def test_valid_starts_refuses_to_span_a_gap():
    """A window must never bridge missing timesteps."""
    times = np.arange(
        np.datetime64("2000-01-01T00"),
        np.datetime64("2000-01-04T00"),
        np.timedelta64(6, "h"),
    )
    gapped = np.delete(times, [5, 6])
    starts = valid_starts(gapped, span=3)
    for s in starts:
        window = gapped[s : s + 3]
        assert (np.diff(window) == np.timedelta64(6, "h")).all()
    assert len(starts) == len(gapped) - 2 - 2  # two windows lost to the gap


def test_sample_shapes(source: WindowSource):
    sample = source.sample(0)
    n_dynamic = len(source.spec.time_channels)
    n_input_channels = n_dynamic + 1 + len(FORCING_CHANNELS)  # +1 static

    assert sample["input"].shape == (2, n_input_channels, 8, 16)
    assert sample["target"].shape == (1, n_dynamic, 8, 16)
    assert sample["input"].dtype == np.float32
    assert len(source.input_channels) == n_input_channels
    assert source.target_channels == source.spec.time_channels


def test_sample_time_is_a_scalar(source: WindowSource):
    """A 0-d timestamp; anything else batches to (B, 1) rather than (B,)."""
    assert np.asarray(source.sample(0)["time"]).shape == ()


def test_target_is_exactly_one_lead_step_after_the_last_input(source: WindowSource):
    """The index arithmetic, pinned against the raw store."""
    raw = xr.open_zarr(source.store, chunks=None)
    sample = source.sample(7)
    last_input_time = np.array(sample["time"]).astype("datetime64[ns]")
    expected_target_time = last_input_time + np.timedelta64(6, "h")

    expected = raw["2m_temperature"].sel(time=expected_target_time).values
    got = source.dynamic_norm.denormalise(sample["target"])[0, 0]
    np.testing.assert_allclose(got, expected, rtol=1e-4)


def test_windows_never_leave_the_split(tmp_path: Path):
    """Every timestep touched, input and target, is inside the period."""
    path, spec = build_store(tmp_path / "store.zarr")
    norm = compute_stats(path, spec, TRAIN, NormaliseSpec(), log=quiet)
    source = WindowSource(
        path, spec, TRAIN, WindowSpec(n_input=2, lead=4, n_target=2), norm
    )

    last = source.sample(len(source) - 1)
    reach = np.array(last["time"]).astype("datetime64[ns]") + np.timedelta64(
        6 * (4 + 1), "h"
    )
    assert reach <= np.datetime64("2000-12-31T18")


def test_static_channels_are_constant_across_time(source: WindowSource):
    sample = source.sample(3)
    static_index = len(source.spec.time_channels)
    np.testing.assert_array_equal(
        sample["input"][0, static_index], sample["input"][1, static_index]
    )


def test_time_encodings_are_cyclic():
    """New Year's Eve and New Year's Day must be adjacent, not a year apart."""
    times = np.array(["2000-12-31T18", "2001-01-01T00"], dtype="datetime64[ns]")
    enc = time_encodings(times)
    distance = np.linalg.norm(enc[0, :2] - enc[1, :2])
    assert distance < 0.05
    assert np.abs(enc).max() <= 1.0


def test_midnight_and_midday_are_opposite():
    times = np.array(["2000-06-01T00", "2000-06-01T12"], dtype="datetime64[ns]")
    enc = time_encodings(times)
    assert enc[0, 3] == pytest.approx(-enc[1, 3], abs=1e-6)


def test_no_target_mode_for_pretraining(tmp_path: Path):
    """Masked-autoencoder pretraining needs inputs only."""
    path, spec = build_store(tmp_path / "store.zarr")
    norm = compute_stats(path, spec, TRAIN, NormaliseSpec(), log=quiet)
    source = WindowSource(path, spec, TRAIN, WindowSpec(n_input=1, n_target=0), norm)
    sample = source.sample(0)
    assert "target" not in sample
    assert sample["input"].shape[0] == 1


def test_in_memory_matches_lazy(tmp_path: Path):
    """The cache must be an optimisation, not a behaviour change."""
    path, spec = build_store(tmp_path / "store.zarr")
    norm = compute_stats(path, spec, TRAIN, NormaliseSpec(), log=quiet)
    lazy = WindowSource(path, spec, TRAIN, WindowSpec(), norm)
    cached = WindowSource(path, spec, TRAIN, WindowSpec(in_memory=True), norm)
    for i in (0, 5, len(lazy) - 1):
        np.testing.assert_array_equal(
            lazy.sample(i)["input"], cached.sample(i)["input"]
        )


def test_normalised_values_match_the_normaliser(source: WindowSource):
    raw = xr.open_zarr(source.store, chunks=None)
    sample = source.sample(0)
    t0 = np.array(sample["time"]).astype("datetime64[ns]")
    expected = source.dynamic_norm.normalise(
        np.stack(
            [
                raw["2m_temperature"].sel(time=t0).values,
                raw["total_precipitation_6hr"].sel(time=t0).values,
                raw["temperature"].sel(time=t0, level=500).values,
                raw["temperature"].sel(time=t0, level=850).values,
            ]
        )
    )
    n_dynamic = len(source.spec.time_channels)
    np.testing.assert_allclose(sample["input"][-1, :n_dynamic], expected, rtol=1e-5)


def test_bad_window_specs_rejected():
    with pytest.raises(ValueError, match="n_input"):
        WindowSpec(n_input=0)
    with pytest.raises(ValueError, match="lead"):
        WindowSpec(lead=0, n_target=1)


def test_normaliser_round_trip_through_a_sample(source: WindowSource):
    """A sample can always be put back into physical units."""
    sample = source.sample(2)
    n_dynamic = len(source.spec.time_channels)
    physical = source.dynamic_norm.denormalise(sample["input"][:, :n_dynamic])
    assert physical[:, 0].mean() > 200  # temperature back in kelvin
    assert (physical[:, 1] >= 0).all()  # precipitation non-negative


def test_isolated_normaliser_subset_ordering(source: WindowSource):
    """Subsetting must preserve the requested order, not the stored order."""
    reversed_names = list(reversed(source.spec.time_channels))
    sub = Normaliser.load(
        source.dynamic_norm.save(Path(source.store).parent / "s.json")
    ).subset(reversed_names)
    assert sub.channels == tuple(reversed_names)
