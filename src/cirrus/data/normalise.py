"""Per-channel normalisation.

Networks train badly when inputs differ in scale by orders of magnitude
(geopotential ~1e5, specific humidity ~1e-3). Every channel is therefore
z-scored: ``(x - mean) / std``.

Precipitation gets a log transform first, ``log1p(max(x, 0) / scale)``,
because raw 6-hourly totals are near zero most of the time with a long right
tail, and z-scored extremes would land at +30 sigma and dominate gradients.

Important for this project: the log transform is for *inputs*. It compresses
the tail by design, so a loss computed in log space quietly down-weights
extremes. The extremes head in Phase 3 takes precipitation in physical units
as its target and lets the GPD-informed loss handle the tail explicitly.

Statistics are:

- **Computed on training years only.** Stats that include test years leak
  information about them.
- **Latitude-weighted** by cos(latitude), because a 5.625 deg cell near the
  pole covers a fraction of the area of one at the equator. This matches the
  latitude-weighted loss used in training.
- **Accumulated in float64.** Summing ~1e8 float32 values loses precision;
  float64 sums of x and x**2 are accurate here even for geopotential.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from cirrus.config import read_yaml_mapping
from cirrus.data.ingest import IngestSpec, check_layout
from cirrus.data.splits import Period

# A channel whose standard deviation is below this is treated as constant.
# Dividing by a near-zero std would turn rounding noise into huge values.
MIN_STD = 1e-12


@dataclass(frozen=True)
class NormaliseSpec:
    """How to normalise, and where to write the statistics."""

    log_variables: tuple[str, ...] = ("total_precipitation_6hr",)
    log_scale: float = 1e-3  # in the variable's units: 1e-3 m = 1 mm
    latitude_weighted: bool = True
    output: str = "data/stats/era5_5625_train.json"

    @classmethod
    def from_yaml(cls, path: str | Path) -> NormaliseSpec:
        """Load a spec, rejecting unknown keys. YAML lists become tuples."""
        raw = read_yaml_mapping(path, (f.name for f in fields(cls)))
        as_tuples: dict[str, Any] = {
            k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()
        }
        return cls(**as_tuples)


@dataclass(frozen=True, eq=False)
class Normaliser:
    """Forward and inverse transforms for an ordered set of channels.

    Arrays passed in must hold channels on axis -3, i.e. ``(..., C, H, W)``,
    in exactly the order of ``channels``.
    """

    channels: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    log_channels: tuple[str, ...]
    log_scale: float
    meta: dict[str, Any]

    @property
    def _log_mask(self) -> np.ndarray:
        return np.array([c in self.log_channels for c in self.channels])

    def _check(self, x: np.ndarray) -> None:
        if x.ndim < 3 or x.shape[-3] != len(self.channels):
            raise ValueError(
                f"expected channels on axis -3 with {len(self.channels)} channels, "
                f"got array of shape {x.shape}"
            )

    def normalise(self, x: np.ndarray) -> np.ndarray:
        """Physical units -> model space."""
        x = np.array(x, dtype=np.float32)  # copy: never mutate the caller's data
        self._check(x)
        mask = self._log_mask
        if mask.any():
            x[..., mask, :, :] = np.log1p(
                np.clip(x[..., mask, :, :], 0.0, None) / self.log_scale
            )
        mean = self.mean.astype(np.float32)[:, None, None]
        std = self.std.astype(np.float32)[:, None, None]
        out: np.ndarray = (x - mean) / std
        return out

    def denormalise(self, z: np.ndarray) -> np.ndarray:
        """Model space -> physical units. Exact inverse for non-negative input.

        Log channels are clipped at zero on the way back: a model output below
        the transform's floor must not become negative precipitation.
        """
        z = np.asarray(z, dtype=np.float32)
        self._check(z)
        mean = self.mean.astype(np.float32)[:, None, None]
        std = self.std.astype(np.float32)[:, None, None]
        x: np.ndarray = z * std + mean
        mask = self._log_mask
        if mask.any():
            x[..., mask, :, :] = np.clip(
                np.expm1(x[..., mask, :, :]) * self.log_scale, 0.0, None
            )
        return x

    def subset(self, names: Sequence[str]) -> Normaliser:
        """Return a normaliser for some of the channels, in the given order."""
        index = [self.channels.index(n) for n in names]
        return Normaliser(
            channels=tuple(names),
            mean=self.mean[index],
            std=self.std[index],
            log_channels=tuple(n for n in names if n in self.log_channels),
            log_scale=self.log_scale,
            meta=self.meta,
        )

    def save(self, path: str | Path) -> Path:
        """Write as human-readable JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "channels": list(self.channels),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "log_channels": list(self.log_channels),
            "log_scale": self.log_scale,
            "meta": self.meta,
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Normaliser:
        """Read what :meth:`save` wrote."""
        payload = json.loads(Path(path).read_text())
        return cls(
            channels=tuple(payload["channels"]),
            mean=np.array(payload["mean"], dtype=np.float64),
            std=np.array(payload["std"], dtype=np.float64),
            log_channels=tuple(payload["log_channels"]),
            log_scale=float(payload["log_scale"]),
            meta=dict(payload["meta"]),
        )


def _latitude_weights(ds: xr.Dataset, weighted: bool) -> np.ndarray:
    """Build a (lat, lon) weight field: cos(latitude), or uniform."""
    lat = ds["latitude"].values
    per_row = np.cos(np.deg2rad(lat)) if weighted else np.ones_like(lat)
    return np.broadcast_to(
        per_row[:, None], (ds.sizes["latitude"], ds.sizes["longitude"])
    ).astype(np.float64)


def _time_channel_arrays(
    block: xr.Dataset, spec: IngestSpec
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (channel name, (time, lat, lon) array) in canonical order."""
    for var in spec.surface_variables:
        yield var, block[var].values
    for var in spec.upper_variables:
        for lev in spec.levels:
            yield f"{var}_{lev}", block[var].sel(level=lev).values


def compute_stats(
    store: str | Path,
    spec: IngestSpec,
    train: Period,
    nspec: NormaliseSpec,
    log: Callable[[str], None] = print,
) -> Normaliser:
    """Weighted mean and std of every channel over the training period.

    Reads one year at a time, so memory stays at a few hundred MB regardless
    of record length.
    """
    ds = xr.open_zarr(store, chunks=None)
    check_layout(ds)
    weights = _latitude_weights(ds, nspec.latitude_weighted)

    time_names = spec.time_channels
    log_channels = tuple(c for c in time_names if c in nspec.log_variables)
    is_log = {c: c in log_channels for c in time_names}

    n = len(time_names)
    sum_w = np.zeros(n)
    sum_wx = np.zeros(n)
    sum_wx2 = np.zeros(n)

    for year in train.years:
        start, end = train.clip_to_year(year)
        block = ds[spec.time_varying].sel(time=slice(start, end)).load()
        n_steps = block.sizes["time"]
        if n_steps == 0:
            continue
        for i, (name, values) in enumerate(_time_channel_arrays(block, spec)):
            x = values.astype(np.float64)
            if is_log[name]:
                x = np.log1p(np.clip(x, 0.0, None) / nspec.log_scale)
            sum_w[i] += weights.sum() * n_steps
            sum_wx[i] += (x * weights).sum()
            sum_wx2[i] += (x * x * weights).sum()
        log(f"{year}: {n_steps} steps accumulated")

    if not (sum_w > 0).all():
        raise ValueError(f"no data in training period {train.start}..{train.end}")

    mean = sum_wx / sum_w
    std = np.sqrt(np.clip(sum_wx2 / sum_w - mean**2, 0.0, None))

    static_mean, static_std = [], []
    for var in spec.static_variables:
        x = ds[var].values.astype(np.float64)
        m = (x * weights).sum() / weights.sum()
        static_mean.append(m)
        static_std.append(np.sqrt(((x - m) ** 2 * weights).sum() / weights.sum()))

    all_mean = np.concatenate([mean, np.array(static_mean)])
    all_std = np.concatenate([std, np.array(static_std)])
    all_std = np.where(all_std < MIN_STD, 1.0, all_std)

    return Normaliser(
        channels=(*time_names, *spec.static_variables),
        mean=all_mean,
        std=all_std,
        log_channels=log_channels,
        log_scale=nspec.log_scale,
        meta={
            "store": str(store),
            "train": [train.start, train.end],
            "latitude_weighted": nspec.latitude_weighted,
        },
    )
