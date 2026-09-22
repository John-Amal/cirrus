"""Pull a subset of ERA5 into a local zarr store.

The source is WeatherBench 2's ERA5, already regridded to 64x32 (5.625 deg)
and 6-hourly with first-order conservative regridding. Using a regridding
done once, upstream, with a documented method matters here: the DestinE
storylines must later be coarsened with the *same* method, or any comparison
of tail statistics between the two is confounded by the regridding itself.

Design choices worth knowing:

- **Validate before downloading.** Every requested variable and level is
  checked against the source first, so a typo fails in seconds rather than
  after an hour of transfer. The same check will verify the ERA5-storyline
  variable intersection later.
- **Year by year, resumable.** Each year is downloaded and appended
  separately. If the process dies at 1997, rerunning continues from 1997.
- **Canonical layout.** Latitude is sorted ascending and dimensions ordered
  (time, level, latitude, longitude) regardless of how the source stores
  them, so nothing downstream has to care.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from cirrus.config import read_yaml_mapping
from cirrus.data.splits import check_date

WB2_ERA5_64X32 = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
)

# Zarr format 2: what WeatherBench uses, with consolidated metadata, and
# readable by every tool in the stack. Format 3 is newer but its consolidated
# metadata is not yet standardised.
ZARR_FORMAT = 2

# Every array in a cirrus store has its dims in this order (skipping any it
# lacks). Downstream code relies on it, and check_layout() verifies it.
CANONICAL_ORDER = ("time", "level", "latitude", "longitude")

# One day of 6-hourly steps. Every year holds a whole number of days, so
# year-sized appends always align with chunk boundaries.
TIME_CHUNK = 4


@dataclass(frozen=True)
class IngestSpec:
    """What to pull, from where, to where."""

    source: str = WB2_ERA5_64X32
    output: str = "data/era5_5625.zarr"
    start: str = "2010-01-01"
    end: str = "2010-01-31"
    surface_variables: tuple[str, ...] = ()
    upper_variables: tuple[str, ...] = ()
    levels: tuple[int, ...] = ()
    static_variables: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject malformed specs before anything touches the network."""
        check_date(self.start, "start")
        check_date(self.end, "end")
        if self.start > self.end:
            raise ValueError(f"start {self.start} is after end {self.end}")
        if self.upper_variables and not self.levels:
            raise ValueError("upper_variables given but no levels")

    @classmethod
    def from_yaml(cls, path: str | Path) -> IngestSpec:
        """Load a spec, rejecting unknown keys. YAML lists become tuples."""
        raw = read_yaml_mapping(path, (f.name for f in fields(cls)))
        as_tuples: dict[str, Any] = {
            k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()
        }
        return cls(**as_tuples)

    @property
    def time_varying(self) -> list[str]:
        """Variables with a time dimension."""
        return [*self.surface_variables, *self.upper_variables]

    @property
    def time_channels(self) -> tuple[str, ...]:
        """Channel names in canonical order, levels unrolled.

        This ordering is a contract: normalisation statistics, dataset tensors
        and model outputs all index channels by it. Upper-air channels are
        named ``<variable>_<level>``, e.g. ``temperature_850``.
        """
        upper = tuple(f"{v}_{lev}" for v in self.upper_variables for lev in self.levels)
        return (*self.surface_variables, *upper)

    @property
    def n_time_channels(self) -> int:
        """Channels per timestep once levels are unrolled into channels."""
        return len(self.time_channels)


def open_source(url: str) -> xr.Dataset:
    """Open a zarr store lazily. Public GCS buckets need anonymous access.

    ``chunks=None`` keeps arrays lazy without requiring dask: nothing is
    transferred until values are actually indexed.
    """
    storage_options = {"token": "anon"} if url.startswith("gs://") else None
    ds: xr.Dataset = xr.open_zarr(url, chunks=None, storage_options=storage_options)
    return ds


def check_available(ds: xr.Dataset, spec: IngestSpec) -> None:
    """Fail fast, and completely, if the source cannot satisfy the spec.

    Reports every problem at once rather than the first one found, so a
    variable-list mismatch is fixed in one edit, not five.
    """
    problems: list[str] = []

    wanted = [*spec.time_varying, *spec.static_variables]
    missing = [v for v in wanted if v not in ds.data_vars]
    if missing:
        problems.append(f"variables not in source: {missing}")

    if spec.upper_variables:
        available = {int(x) for x in ds["level"].values}
        missing_levels = [lev for lev in spec.levels if lev not in available]
        if missing_levels:
            problems.append(
                f"levels not in source: {missing_levels} "
                f"(available: {sorted(available)})"
            )

    first = str(ds["time"].values[0])[:10]
    last = str(ds["time"].values[-1])[:10]
    if spec.start < first or spec.end > last:
        problems.append(
            f"requested {spec.start}..{spec.end}, source covers {first}..{last}"
        )

    if problems:
        raise ValueError(f"{spec.source}:\n  " + "\n  ".join(problems))


def _canonical(ds: xr.Dataset) -> xr.Dataset:
    """Ascending latitude, fixed dim order, float32, source encoding dropped.

    Dropping encoding matters: arrays opened from a remote zarr carry that
    store's chunking and compression settings, which conflict with ours on
    write and produce confusing errors.
    """
    ds = ds.sortby("latitude")
    # Name every dim explicitly. Pinning only some of them and passing `...`
    # leaves the rest in the source's order -- and WeatherBench stores
    # longitude before latitude.
    ds = ds.transpose(*[d for d in CANONICAL_ORDER if d in ds.dims])
    ds = ds.astype("float32")
    for name in ds.variables:
        ds[name].encoding = {}
    return ds


def check_layout(ds: xr.Dataset) -> None:
    """Raise if a store is not in cirrus's canonical layout.

    Consumers call this on open, so a store written by older or foreign code
    fails with a clear message instead of a broadcast error deep in the maths
    -- or worse, silently, if the grid happens to be square.
    """
    problems = []
    for name, da in ds.data_vars.items():
        expected = tuple(d for d in CANONICAL_ORDER if d in da.dims)
        if da.dims != expected:
            problems.append(f"{name}: dims {da.dims}, expected {expected}")
    if "latitude" in ds.coords and not (ds["latitude"].diff("latitude") > 0).all():
        problems.append("latitude is not strictly ascending")
    if problems:
        raise ValueError(
            "store is not in canonical layout; re-run `cirrus ingest` into a "
            "fresh output:\n  " + "\n  ".join(problems)
        )


def select_time_varying(
    ds: xr.Dataset, spec: IngestSpec, start: str, end: str
) -> xr.Dataset:
    """Subset variables, levels and a time window. Still lazy."""
    out = ds[spec.time_varying].sel(time=slice(start, end))
    if spec.upper_variables:
        out = out.sel(level=list(spec.levels))
    return _canonical(out)


def select_static(ds: xr.Dataset, spec: IngestSpec) -> xr.Dataset:
    """Time-invariant fields such as orography and the land-sea mask."""
    return _canonical(ds[list(spec.static_variables)])


def estimate_bytes(ds: xr.Dataset, spec: IngestSpec) -> int:
    """Size of the requested subset on disk, uncompressed, in bytes."""
    n_time = ds["time"].sel(time=slice(spec.start, spec.end)).size
    grid = ds.sizes["latitude"] * ds.sizes["longitude"]
    time_varying = n_time * spec.n_time_channels * grid
    static = len(spec.static_variables) * grid
    return 4 * (time_varying + static)


def describe(ds: xr.Dataset, spec: IngestSpec) -> str:
    """Human-readable summary of what an ingest would do. For --dry-run."""
    n_time = ds["time"].sel(time=slice(spec.start, spec.end)).size
    return "\n".join(
        [
            f"source:    {spec.source}",
            f"output:    {spec.output}",
            f"period:    {spec.start} .. {spec.end} ({n_time:,} timesteps)",
            f"grid:      {ds.sizes['latitude']} x {ds.sizes['longitude']}",
            f"surface:   {', '.join(spec.surface_variables) or '-'}",
            f"upper:     {', '.join(spec.upper_variables) or '-'}",
            f"levels:    {', '.join(map(str, spec.levels)) or '-'}",
            f"static:    {', '.join(spec.static_variables) or '-'}",
            f"channels:  {spec.n_time_channels} time-varying "
            f"+ {len(spec.static_variables)} static",
            f"size:      ~{estimate_bytes(ds, spec) / 1e9:.2f} GB uncompressed",
        ]
    )


def _last_written(output: Path) -> np.datetime64 | None:
    """Most recent timestep already on disk, or None for a fresh store."""
    if not output.exists():
        return None
    existing = xr.open_zarr(output, chunks=None)
    last: np.datetime64 = existing["time"].values[-1]
    return last


def _encoding(ds: xr.Dataset, static: list[str]) -> dict[str, dict[str, Any]]:
    """Chunk layout for the first write: one day per chunk, full grid."""
    encoding: dict[str, dict[str, Any]] = {}
    for name in ds.data_vars:
        shape = ds[name].shape
        chunks = shape if name in static else (TIME_CHUNK, *shape[1:])
        encoding[str(name)] = {"chunks": chunks}
    return encoding


def ingest(
    spec: IngestSpec,
    *,
    source: xr.Dataset | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Download the spec'd subset to ``spec.output``, resuming if interrupted.

    Args:
        spec: What to pull.
        source: An already-open dataset; used by tests to avoid the network.
        log: Where progress lines go.

    Returns:
        Path to the local zarr store.

    """
    ds = open_source(spec.source) if source is None else source
    check_available(ds, spec)

    output = Path(spec.output)
    last = _last_written(output)
    if last is not None:
        log(f"resuming: store already holds data up to {str(last)[:16]}")

    for year in range(int(spec.start[:4]), int(spec.end[:4]) + 1):
        window_start = max(spec.start, f"{year}-01-01")
        window_end = min(spec.end, f"{year}-12-31")
        chunk = select_time_varying(ds, spec, window_start, window_end)

        if last is not None:
            keep = np.flatnonzero(chunk["time"].values > last)
            chunk = chunk.isel(time=keep)
        if chunk.sizes["time"] == 0:
            continue

        started = time.perf_counter()
        chunk = chunk.load()  # this line is the actual download

        if output.exists():
            chunk.to_zarr(output, mode="a", append_dim="time", zarr_format=ZARR_FORMAT)
        else:
            static = select_static(ds, spec).load()
            first = xr.merge([chunk, static])
            first.to_zarr(
                output,
                mode="w",
                zarr_format=ZARR_FORMAT,
                encoding=_encoding(first, list(spec.static_variables)),
            )

        megabytes = chunk.nbytes / 1e6
        seconds = time.perf_counter() - started
        log(
            f"{year}: {chunk.sizes['time']:>5} steps  "
            f"{megabytes:7.1f} MB  {seconds:6.1f}s"
        )

    return output
