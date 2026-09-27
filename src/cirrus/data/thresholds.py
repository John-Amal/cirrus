"""Per-cell exceedance thresholds for precipitation.

The tail-aware objective needs a threshold above which errors are scored, and
the Phase 4 return-level estimates need one above which a generalised Pareto
is fitted. It is the same threshold, and it has to satisfy both.

A fixed quantile of all timesteps does not: at 5.625 degrees and 6-hourly
resolution much of the subtropics is dry well past the 95th percentile, so
the threshold would be zero there. ``max(x, 0)`` is the identity, which means
the tail weighting would silently do nothing in exactly the regions where a
model could most easily look good. Zero-valued thresholds would also leave a
GPD fit with no exceedances to fit to.

Instead the threshold is set **per cell so that a fixed fraction of
timesteps exceeds it**, subject to a floor. Every cell then contributes a
comparable number of exceedances, which is what makes per-cell tail
statistics comparable across the globe, and no cell gets a threshold of zero.

Computed on **training years only**. A threshold derived from the whole
record would leak information about the test period into training, exactly as
normalisation statistics would.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from cirrus.config import read_yaml_mapping
from cirrus.data.ingest import IngestSpec, check_layout
from cirrus.data.splits import Period


@dataclass(frozen=True)
class ThresholdSpec:
    """How the exceedance threshold is defined."""

    variable: str = "total_precipitation_6hr"
    exceedance_rate: float = 0.02  # 2% of timesteps, ~1050 events per cell
    floor: float = 0.1  # mm; no cell may end up with a zero threshold
    units_scale: float = 1000.0  # ERA5 stores precipitation in metres
    output: str = "data/stats/thresholds_train.json"

    def __post_init__(self) -> None:
        """Reject rates that would leave nothing, or everything, in the tail."""
        if not 0.0 < self.exceedance_rate < 0.5:
            raise ValueError(
                f"exceedance_rate must be in (0, 0.5), got {self.exceedance_rate}"
            )
        if self.floor < 0:
            raise ValueError("floor must not be negative")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ThresholdSpec:
        """Load a spec, rejecting unknown keys."""
        raw: dict[str, Any] = read_yaml_mapping(path, (f.name for f in fields(cls)))
        return cls(**raw)


@dataclass
class Thresholds:
    """Per-cell thresholds in physical units, with how they were made."""

    values: np.ndarray  # (lat, lon), mm
    meta: dict[str, Any]

    @property
    def floored_fraction(self) -> float:
        """Share of cells sitting on the floor rather than on their quantile."""
        return float(self.meta.get("floored_fraction", float("nan")))

    def save(self, path: str | Path) -> Path:
        """Write as JSON, so the thresholds are inspectable and diffable."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"values": self.values.tolist(), "meta": self.meta}, indent=2)
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Thresholds:
        """Read what :meth:`save` wrote."""
        payload = json.loads(Path(path).read_text())
        return cls(
            values=np.array(payload["values"], dtype=np.float32),
            meta=dict(payload["meta"]),
        )


def compute_thresholds(
    store: str | Path,
    data_spec: IngestSpec,
    period: Period,
    spec: ThresholdSpec,
    log: Callable[[str], None] = print,
) -> Thresholds:
    """Per-cell thresholds at a fixed exceedance rate over ``period``.

    Loads the whole training record for one variable -- about 430 MB at
    5.625 degrees -- because a quantile needs all the values at once.
    """
    dataset = xr.open_zarr(store, chunks=None)
    check_layout(dataset)
    if spec.variable not in dataset.data_vars:
        raise ValueError(f"{spec.variable} is not in {store}")

    selected = dataset[spec.variable].sel(time=slice(period.start, period.end))
    n_steps = int(selected.sizes["time"])
    if n_steps == 0:
        raise ValueError(f"no data in {period.start}..{period.end}")
    log(f"loading {n_steps:,} steps of {spec.variable}")

    # Clip first: conservative regridding leaves tiny negative residue, and a
    # negative value must not count toward the dry end of the distribution.
    values = np.clip(selected.values.astype(np.float32), 0.0, None) * spec.units_scale

    quantile = float(1.0 - spec.exceedance_rate)
    raw = np.quantile(values, quantile, axis=0).astype(np.float32)
    thresholds = np.maximum(raw, spec.floor).astype(np.float32)

    exceedances = (values > thresholds[None]).mean(axis=0)
    floored = float((raw < spec.floor).mean())

    meta = {
        "variable": spec.variable,
        "units": "mm",
        "exceedance_rate": spec.exceedance_rate,
        "floor": spec.floor,
        "period": [period.start, period.end],
        "n_steps": n_steps,
        "floored_fraction": floored,
        "achieved_rate_median": float(np.median(exceedances)),
        "events_per_cell_median": float(np.median(exceedances) * n_steps),
    }

    log(
        f"threshold: median {float(np.median(thresholds)):.2f} mm, "
        f"range {float(thresholds.min()):.2f}-{float(thresholds.max()):.2f} mm"
    )
    log(
        f"{floored:.1%} of cells sit on the {spec.floor} mm floor "
        f"(too dry to reach the {quantile:.1%} quantile)"
    )
    log(
        f"median {meta['events_per_cell_median']:.0f} exceedances per cell "
        f"over {n_steps:,} steps"
    )
    return Thresholds(values=thresholds, meta=meta)
