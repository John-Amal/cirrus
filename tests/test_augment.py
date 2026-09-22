"""Tests for the longitude roll.

The interesting one is ``test_roll_preserves_local_solar_time``. The roll is
only valid if the time encodings move with it: otherwise the augmentation
teaches the model that the sun rises at a different clock time, which is
worse than no augmentation at all. That test builds a field defined purely by
local solar time and checks the relationship survives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from test_windows import TRAIN, build_store, quiet

from cirrus.data.ingest import IngestSpec
from cirrus.data.normalise import NormaliseSpec, compute_stats
from cirrus.data.splits import Period
from cirrus.data.windows import AugmentSpec, WindowSource, WindowSpec

N_LON = 16
PERIOD = Period("2000-01-01", "2000-12-31")


def solar_store(path: Path) -> tuple[str, IngestSpec]:
    """Build a store whose temperature depends only on local solar time.

    ``T(lon, t) = cos(2 pi (utc_hours / 24 + lon / 360))`` is constant along
    lines of constant local solar time, which is exactly the symmetry the
    roll has to respect.
    """
    times = np.arange(
        np.datetime64("2000-01-01T00"),
        np.datetime64("2000-03-01T00"),
        np.timedelta64(6, "h"),
    ).astype("datetime64[ns]")
    lat = np.linspace(-80.0, 80.0, 8)
    lon = np.arange(N_LON) * (360.0 / N_LON)

    day_start = times.astype("datetime64[D]").astype("datetime64[ns]")
    utc_hours = (times - day_start) / np.timedelta64(1, "h")
    phase = utc_hours[:, None] / 24.0 + lon[None, :] / 360.0
    field = np.cos(2 * np.pi * phase)  # (time, lon)
    t2m = np.broadcast_to(field[:, None, :], (len(times), 8, N_LON))

    ds = xr.Dataset(
        {"2m_temperature": (("time", "latitude", "longitude"), t2m.copy())},
        coords={"time": times, "latitude": lat, "longitude": lon},
    ).astype("float32")
    ds.to_zarr(path, zarr_format=2)

    spec = IngestSpec(
        output=str(path),
        start="2000-01-01",
        end="2000-02-29",
        surface_variables=("2m_temperature",),
    )
    return str(path), spec


def make_source(tmp_path: Path, augment: AugmentSpec | None) -> WindowSource:
    path, spec = build_store(tmp_path / "store.zarr")
    norm = compute_stats(path, spec, TRAIN, NormaliseSpec(), log=quiet)
    return WindowSource(path, spec, TRAIN, WindowSpec(), norm, augment=augment)


def test_no_augmentation_means_no_shift(tmp_path: Path):
    source = make_source(tmp_path, None)
    assert all(source.roll_shift(i) == 0 for i in range(20))


def test_shifts_are_reproducible_and_varied(tmp_path: Path):
    """Same seed gives the same draw; draws still cover the grid."""
    a = make_source(tmp_path / "a", AugmentSpec(seed=7))
    b = make_source(tmp_path / "b", AugmentSpec(seed=7))
    shifts = [a.roll_shift(i) for i in range(200)]
    assert shifts == [b.roll_shift(i) for i in range(200)]
    assert len(set(shifts)) > 8  # not stuck on one value
    assert all(0 <= s < a.n_longitude for s in shifts)


def test_epoch_changes_the_draw(tmp_path: Path):
    source = make_source(tmp_path, AugmentSpec(seed=0))
    first = [source.roll_shift(i) for i in range(50)]
    source.set_epoch(1)
    assert first != [source.roll_shift(i) for i in range(50)]


def spatial(source: WindowSource, sample: dict[str, np.ndarray]) -> np.ndarray:
    """Dynamic and static channels only.

    The forcing channels are excluded on purpose: they are *supposed* to
    differ between a plain and a rolled sample, because the roll moves the
    clock. Comparing them would be comparing the feature to itself.
    """
    n_spatial = len(source.spec.time_channels) + len(source.spec.static_variables)
    return sample["input"][:, :n_spatial]


def test_roll_is_a_permutation_not_a_distortion(tmp_path: Path):
    """Rolling reorders values; it must not create or destroy any."""
    plain = make_source(tmp_path / "p", None)
    rolled = make_source(tmp_path / "r", AugmentSpec(seed=3))
    index = next(i for i in range(50) if rolled.roll_shift(i) != 0)

    a = np.sort(spatial(plain, plain.sample(index)).ravel())
    b = np.sort(spatial(rolled, rolled.sample(index)).ravel())
    np.testing.assert_allclose(a, b, rtol=1e-6)


def test_input_and_target_roll_together(tmp_path: Path):
    """A different shift for input and target would be silently corrupting."""
    plain = make_source(tmp_path / "p", None)
    rolled = make_source(tmp_path / "r", AugmentSpec(seed=3))
    index = next(i for i in range(50) if rolled.roll_shift(i) != 0)
    shift = rolled.roll_shift(index)

    expected = np.roll(plain.sample(index)["target"], shift, axis=-1)
    np.testing.assert_allclose(rolled.sample(index)["target"], expected, rtol=1e-6)


def test_latitude_is_untouched(tmp_path: Path):
    """The roll must not move data between latitudes."""
    plain = make_source(tmp_path / "p", None)
    rolled = make_source(tmp_path / "r", AugmentSpec(seed=3))
    index = next(i for i in range(50) if rolled.roll_shift(i) != 0)
    a = spatial(plain, plain.sample(index)).mean(axis=-1)  # zonal mean per lat
    b = spatial(rolled, rolled.sample(index)).mean(axis=-1)
    np.testing.assert_allclose(a, b, rtol=1e-5)


def test_roll_preserves_local_solar_time(tmp_path: Path):
    """The physics check: rolled fields must match the shifted clock.

    If the time encodings did not move with the roll, the augmented sample
    would show the diurnal wave in the wrong place relative to the hour the
    model is given.
    """
    path, spec = solar_store(tmp_path / "solar.zarr")
    norm = compute_stats(path, spec, PERIOD, NormaliseSpec(), log=quiet)
    source = WindowSource(
        path, spec, PERIOD, WindowSpec(), norm, augment=AugmentSpec(seed=5)
    )
    index = next(i for i in range(60) if source.roll_shift(i) != 0)
    shift = source.roll_shift(index)
    sample = source.sample(index)

    # The field the model was given, back in physical units, against the
    # clock it was given alongside it.
    field = source.dynamic_norm.denormalise(sample["input"][:, :1])[-1, 0, 0]
    lon = np.arange(N_LON) * (360.0 / N_LON)
    consistent = np.cos(2 * np.pi * (told_hours(sample) / 24.0 + lon / 360.0))

    np.testing.assert_allclose(field, consistent, atol=1e-4)
    assert shift != 0  # the test would be vacuous otherwise


def told_hours(sample: dict[str, np.ndarray]) -> float:
    """Recover the UTC hour the model was given, from its own sin/cos."""
    sin_hour = float(sample["input"][-1, -2, 0, 0])
    cos_hour = float(sample["input"][-1, -1, 0, 0])
    return float((np.arctan2(sin_hour, cos_hour) / (2 * np.pi) * 24) % 24)


def test_wrong_sign_would_fail_this_test(tmp_path: Path):
    """Guard the guard: the unshifted clock must disagree with the field.

    Derives both hours from the forcing channels, so there is no datetime
    arithmetic to get wrong -- which is how the first version of this test
    managed to fail for an unrelated reason.
    """
    path, spec = solar_store(tmp_path / "solar.zarr")
    norm = compute_stats(path, spec, PERIOD, NormaliseSpec(), log=quiet)
    source = WindowSource(
        path, spec, PERIOD, WindowSpec(), norm, augment=AugmentSpec(seed=5)
    )
    index = next(i for i in range(60) if source.roll_shift(i) != 0)
    shift = source.roll_shift(index)
    sample = source.sample(index)

    field = source.dynamic_norm.denormalise(sample["input"][:, :1])[-1, 0, 0]
    # The clock before the roll shifted it: what a forgotten pairing would use.
    unshifted = (told_hours(sample) + shift * 24.0 / N_LON) % 24
    lon = np.arange(N_LON) * (360.0 / N_LON)
    wrong = np.cos(2 * np.pi * (unshifted / 24.0 + lon / 360.0))

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(field, wrong, atol=1e-3)
