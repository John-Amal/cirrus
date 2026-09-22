"""Shared fixtures.

pytest imports this automatically and makes its fixtures available to every
test module. That is the point: no test file needs to import another, which
keeps import-sorting deterministic and removes a genuine smell.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from cirrus.data.ingest import IngestSpec

StoreBuilder = Callable[..., tuple[str, IngestSpec]]


@pytest.fixture
def quiet() -> Callable[[str], None]:
    """Swallow progress logging."""

    def _quiet(_: str) -> None:
        pass

    return _quiet


@pytest.fixture
def build_store() -> StoreBuilder:
    """Return a builder for a small store in cirrus's canonical layout.

    ``drop`` removes timesteps, to exercise gap handling.
    """

    def _build(path: Path, drop: slice | None = None) -> tuple[str, IngestSpec]:
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

    return _build
