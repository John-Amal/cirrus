"""Turning a zarr store into training samples.

All of the real logic lives here, in numpy, with no torch import. The
PyTorch ``Dataset`` in :mod:`cirrus.data.dataset` is a thin wrapper. Keeping
them apart means this — the part with the off-by-one errors in it — can be
tested without a GPU, a model, or torch at all.

A sample is:

- **input**: ``n_input`` consecutive states ending at time t, as
  ``(n_input, C_in, H, W)``. ``C_in`` is the dynamic channels, then the
  static fields, then the time encodings.
- **target**: ``n_target`` states starting ``lead`` steps after t, as
  ``(n_target, C_dynamic, H, W)``. Targets carry dynamic channels only —
  statics are constant and forcings are known in advance, so predicting
  them would be free marks.

Two input steps is the default because one snapshot carries no tendency
information: a single field cannot distinguish a deepening low from a
filling one.

Two boundaries are enforced, both of which are silent-corruption bugs if
missed:

- **Split boundaries.** Every timestep a sample touches, input *and* target,
  must lie inside the requested period. A window straddling the train/test
  boundary would leak test data into training.
- **Time gaps.** Windows are only valid where timesteps are consecutive at
  the store's own spacing. A window spanning a gap would silently present
  a 6-hour forecast problem as if it were 6-hourly when it is not.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from cirrus.config import read_yaml_mapping
from cirrus.data.ingest import IngestSpec, check_layout
from cirrus.data.normalise import Normaliser
from cirrus.data.splits import Period

FORCING_CHANNELS = ("sin_dayofyear", "cos_dayofyear", "sin_hour", "cos_hour")

NS_PER_DAY = 86_400_000_000_000


@dataclass(frozen=True)
class AugmentSpec:
    """Augmentation settings. Training splits only.

    Only one augmentation is implemented, deliberately.

    The **longitude roll** shifts every field east by k cells. It is valid
    because atmospheric dynamics are equivariant under rotation about the
    Earth's axis: latitude is preserved, so the Coriolis parameter, the
    land-sea contrast relative to latitude, and the jet structure are all
    untouched. It is also free, being a permutation of data already loaded.

    What is deliberately absent:

    - **Mixup** would average two atmospheric states. The average of two
      fields is smoother than either, so mixup systematically weakens
      extremes -- the exact quantity this project measures.
    - **Masking** is not an augmentation here; it is the pretraining
      objective, and belongs in the model (Phase 2).
    - **Latitude flips and rotations** break the physics: they reverse the
      sign of the Coriolis parameter without reversing the flow.
    """

    roll: bool = True
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> AugmentSpec:
        """Load a spec, rejecting unknown keys."""
        raw: dict[str, Any] = read_yaml_mapping(path, (f.name for f in fields(cls)))
        return cls(**raw)


@dataclass(frozen=True)
class WindowSpec:
    """Shape of a training sample."""

    n_input: int = 2
    lead: int = 1
    n_target: int = 1
    forcings: bool = True
    in_memory: bool = False

    def __post_init__(self) -> None:
        """Reject shapes that cannot describe a forecast."""
        if self.n_input < 1:
            raise ValueError("n_input must be at least 1")
        if self.n_target < 0:
            raise ValueError("n_target cannot be negative")
        if self.n_target > 0 and self.lead < 1:
            raise ValueError("lead must be at least 1 when predicting a target")

    @classmethod
    def from_yaml(cls, path: str | Path) -> WindowSpec:
        """Load a spec, rejecting unknown keys."""
        raw: dict[str, Any] = read_yaml_mapping(path, (f.name for f in fields(cls)))
        return cls(**raw)

    @property
    def span(self) -> int:
        """Timesteps from the first input step to the last target step."""
        if self.n_target == 0:
            return self.n_input
        return self.n_input + self.lead - 1 + self.n_target


def time_encodings(times: np.ndarray) -> np.ndarray:
    """Sine and cosine of day-of-year and hour, as ``(T, 4)``.

    Position in the year and in the day are cyclic: 31 December is adjacent
    to 1 January, and 23:00 to 00:00. A raw day number would tell the model
    those are 364 days apart. The sine/cosine pair encodes them as a circle.

    Using the year's own length handles leap years without special cases.
    """
    year_start = times.astype("datetime64[Y]").astype("datetime64[ns]")
    # Explicit unit: adding a bare integer to a datetime64 array is deprecated.
    next_year = (times.astype("datetime64[Y]") + np.timedelta64(1, "Y")).astype(
        "datetime64[ns]"
    )
    year_fraction = (times - year_start) / (next_year - year_start)

    day_start = times.astype("datetime64[D]").astype("datetime64[ns]")
    day_fraction = (times - day_start) / np.timedelta64(1, "D")

    return np.stack(
        [
            np.sin(2 * np.pi * year_fraction),
            np.cos(2 * np.pi * year_fraction),
            np.sin(2 * np.pi * day_fraction),
            np.cos(2 * np.pi * day_fraction),
        ],
        axis=-1,
    ).astype(np.float32)


def valid_starts(times: np.ndarray, span: int) -> np.ndarray:
    """First indices of every run of ``span`` consecutive, evenly spaced steps.

    ``times`` must already be restricted to one split. The spacing is taken
    from the most common gap in the series, so this works for any cadence.
    """
    if len(times) < span:
        return np.empty(0, dtype=np.int64)

    gaps = np.diff(times)
    step = np.median(gaps).astype(gaps.dtype)
    contiguous = gaps == step

    # A window starting at i is valid when the span - 1 gaps after i are all
    # the nominal step. Cumulative sums give that in one pass.
    ok = np.concatenate([contiguous, [False]])
    run = np.cumsum(np.concatenate([[0], ok.astype(np.int64)]))
    n_starts = len(times) - span + 1
    full = run[np.arange(n_starts) + span - 1] - run[np.arange(n_starts)]
    return np.flatnonzero(full == span - 1).astype(np.int64)


class WindowSource:
    """Samples drawn from one store, one split, one window shape.

    Opening is lazy: the zarr handle is created on first use, not in
    ``__init__``. PyTorch dataloader workers are separate processes, and an
    open handle created in the parent would have to be pickled across to
    them. Opening per worker avoids that entirely.
    """

    def __init__(
        self,
        store: str | Path,
        spec: IngestSpec,
        period: Period,
        window: WindowSpec,
        normaliser: Normaliser,
        augment: AugmentSpec | None = None,
    ) -> None:
        self.store = str(store)
        self.augment = augment
        self.epoch = 0
        self.spec = spec
        self.period = period
        self.window = window
        self.dynamic_norm = normaliser.subset(spec.time_channels)
        self.static_norm = (
            normaliser.subset(list(spec.static_variables))
            if spec.static_variables
            else None
        )
        self._ds: xr.Dataset | None = None
        self._static: np.ndarray | None = None
        self._cache: np.ndarray | None = None

        ds = self._open()
        self.times: np.ndarray = ds["time"].values
        self.starts = valid_starts(self.times, window.span)
        self.n_longitude = int(ds.sizes["longitude"])

    def set_epoch(self, epoch: int) -> None:
        """Change the augmentation draw between epochs."""
        self.epoch = epoch

    def roll_shift(self, index: int) -> int:
        """Longitude shift in grid cells for this sample, 0 if not augmenting.

        Drawn statelessly from (seed, epoch, index) rather than from a live
        RNG. Dataloader workers are forked or spawned copies: a stateful
        numpy generator would be duplicated, and every worker would draw the
        same sequence -- a classic and near-invisible PyTorch bug. A pure
        function of the index cannot have that problem, and makes runs exactly
        reproducible.
        """
        if self.augment is None or not self.augment.roll:
            return 0
        rng = np.random.default_rng((self.augment.seed, self.epoch, index))
        return int(rng.integers(self.n_longitude))

    def _open(self) -> xr.Dataset:
        if self._ds is None:
            ds = xr.open_zarr(self.store, chunks=None)
            check_layout(ds)
            self._ds = ds.sel(time=slice(self.period.start, self.period.end))
        return self._ds

    @property
    def input_channels(self) -> tuple[str, ...]:
        """Channel names of the input tensor, in order."""
        names = (*self.spec.time_channels, *self.spec.static_variables)
        return (*names, *FORCING_CHANNELS) if self.window.forcings else names

    @property
    def target_channels(self) -> tuple[str, ...]:
        """Channel names of the target tensor, in order."""
        return self.spec.time_channels

    def _dynamic(self, start: int, stop: int) -> np.ndarray:
        """Normalised dynamic channels for ``times[start:stop]``."""
        if self.window.in_memory:
            if self._cache is None:
                self._cache = self._read_dynamic(0, len(self.times))
            return self._cache[start:stop]
        return self._read_dynamic(start, stop)

    def _read_dynamic(self, start: int, stop: int) -> np.ndarray:
        ds = self._open().isel(time=slice(start, stop))
        arrays = [ds[v].values for v in self.spec.surface_variables]
        arrays += [
            ds[v].sel(level=lev).values
            for v in self.spec.upper_variables
            for lev in self.spec.levels
        ]
        stacked = np.stack(arrays, axis=1)  # (time, channel, lat, lon)
        return self.dynamic_norm.normalise(stacked)

    def _statics(self) -> np.ndarray:
        """Normalised static fields, ``(C_static, H, W)``. Read once."""
        if self._static is None:
            ds = self._open()
            stacked = np.stack(
                [ds[v].values for v in self.spec.static_variables], axis=0
            )
            assert self.static_norm is not None
            self._static = self.static_norm.normalise(stacked)
        return self._static

    def __len__(self) -> int:
        """Return the number of valid samples in this split."""
        return len(self.starts)

    def sample(self, index: int) -> dict[str, np.ndarray]:
        """Build one sample as numpy arrays."""
        start = int(self.starts[index])
        w = self.window
        shift = self.roll_shift(index)

        inputs = self._dynamic(start, start + w.n_input)
        if shift:
            inputs = np.roll(inputs, shift, axis=-1)
        n_time, _, height, width = inputs.shape

        extras = []
        if self.spec.static_variables:
            statics = self._statics()
            if shift:
                statics = np.roll(statics, shift, axis=-1)
            extras.append(np.broadcast_to(statics, (n_time, *statics.shape)))
        if w.forcings:
            # Rolling the world east by k cells moves a feature from longitude
            # L to L + k*dL. Its local solar time is only preserved if the UTC
            # the model is told moves back by k * 24/n_longitude hours. Shift
            # the timestamps and recompute, so day-of-year stays consistent
            # when the shift crosses midnight.
            offset = np.timedelta64(round(shift * NS_PER_DAY / self.n_longitude), "ns")
            enc = time_encodings(self.times[start : start + w.n_input] - offset)
            extras.append(
                np.broadcast_to(
                    enc[:, :, None, None], (n_time, enc.shape[1], height, width)
                )
            )
        if extras:
            inputs = np.concatenate([inputs, *extras], axis=1)

        out: dict[str, np.ndarray] = {
            "input": np.ascontiguousarray(inputs, dtype=np.float32),
            "time": self.times[start + w.n_input - 1]
            .astype("datetime64[ns]")
            .view("int64"),
        }
        if w.n_target > 0:
            first = start + w.n_input - 1 + w.lead
            target = self._dynamic(first, first + w.n_target)
            if shift:
                target = np.roll(target, shift, axis=-1)
            out["target"] = np.ascontiguousarray(target, dtype=np.float32)
        return out
