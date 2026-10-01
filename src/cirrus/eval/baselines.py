"""Reference forecasts to measure the trained arms against.

Two baselines, chosen because they fail in opposite ways and together they
bracket what a result means.

**Persistence** predicts that the next six hours look like the last six.
Its implied climatology is the observed one shifted in time, so its tail
amplitude and return levels are right *by construction* while its forecast
skill is poor. That makes it the guard against a tempting misreading of the
Phase 4 result: reproducing the observed tail is not by itself evidence of
anything, because the dumbest possible forecast does it perfectly. Only
reproducing the tail *and* scoring well probabilistically means something.

**Climatology** ignores the input entirely and samples from what this cell
does in this month, historically. It is the conventional no-skill reference:
a model that cannot beat it has learned nothing about the atmosphere, only
about geography and season.

The climatological distribution is stored as per-cell, per-month quantiles
rather than as raw values. The quantile levels are deliberately dense in the
upper tail -- a uniformly spaced table would truncate exactly the part of the
distribution this project is about.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr

from cirrus.data.ingest import IngestSpec, check_layout
from cirrus.data.splits import Period

# Dense through the bulk, then logarithmically dense toward 1 so the far tail
# is resolved: the last level is 1 - 1e-6.
#
# Levels alone cannot lift the ceiling past the training-period maximum for
# that cell and month -- a quantile of a finite sample is bounded by its
# largest member. Extending them resolves the approach to that maximum; it
# does not let a climatological sample exceed anything ever observed.
QUANTILE_LEVELS = np.concatenate(
    [
        np.linspace(0.0005, 0.99, 400),
        1.0 - np.logspace(-2, -6, 100),
    ]
)


@dataclass
class Climatology:
    """Per-cell, per-month quantiles of training-period precipitation."""

    quantiles: np.ndarray  # (12, n_cells, n_levels), mm
    levels: np.ndarray  # (n_levels,)
    meta: dict[str, Any]

    @property
    def n_cells(self) -> int:
        """Number of grid cells, flattened row-major."""
        return int(self.quantiles.shape[1])

    def save(self, path: str | Path) -> Path:
        """Write quantiles as npz and the metadata beside it as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, quantiles=self.quantiles, levels=self.levels)
        path.with_suffix(".json").write_text(json.dumps(self.meta, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Climatology:
        """Read what :meth:`save` wrote."""
        path = Path(path)
        payload = np.load(path)
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return cls(payload["quantiles"], payload["levels"], meta)


def build_climatology(
    store: str | Path,
    data_spec: IngestSpec,
    period: Period,
    variable: str = "total_precipitation_6hr",
    units_scale: float = 1000.0,
) -> Climatology:
    """Per-cell, per-month quantiles over the training period.

    Monthly rather than daily: a day-of-year window at 36 years gives ~4,400
    values per cell, which is enough, but months are simpler and the seasonal
    cycle at 5.625 degrees is not sharp enough to need finer resolution.
    """
    dataset = xr.open_zarr(store, chunks=None)
    check_layout(dataset)
    selected = dataset[variable].sel(time=slice(period.start, period.end))
    values = np.clip(selected.values.astype(np.float32), 0.0, None) * units_scale
    months = selected["time"].dt.month.values

    n_time = values.shape[0]
    grid = values.shape[1:]
    flat = values.reshape(n_time, -1)
    quantiles = np.zeros((12, flat.shape[1], len(QUANTILE_LEVELS)), dtype=np.float32)

    for month in range(1, 13):
        rows = flat[months == month]
        if rows.size == 0:
            raise ValueError(f"no training data in month {month}")
        quantiles[month - 1] = np.quantile(rows, QUANTILE_LEVELS, axis=0).T

    return Climatology(
        quantiles=quantiles,
        levels=QUANTILE_LEVELS.astype(np.float32),
        meta={
            "variable": variable,
            "units": "mm",
            "period": [period.start, period.end],
            "n_steps": int(n_time),
            "grid": list(grid),
            "n_levels": len(QUANTILE_LEVELS),
        },
    )


class ClimatologyBaseline:
    """Samples from the historical distribution for this cell and month.

    Draws invert the stored quantile table: a uniform probability is mapped
    to the level containing it, and that level's value is returned. Cells are
    drawn
    independently, so a sample has no spatial coherence -- which is correct
    for a climatological reference and worth remembering when looking at one
    plotted as a field.
    """

    def __init__(self, climatology: Climatology, device: torch.device) -> None:
        self.table = torch.as_tensor(climatology.quantiles, device=device)
        self.levels = torch.as_tensor(climatology.levels, device=device)
        self.n_levels = int(self.table.shape[-1])
        self.n_cells = climatology.n_cells

    def samples(
        self, times_ns: torch.Tensor, grid: tuple[int, int], draws: int
    ) -> torch.Tensor:
        """``(B, H, W, draws)`` drawn from the right month for each sample."""
        months = months_from_nanoseconds(times_ns).to(self.table.device)
        batch = months.shape[0]

        # Draw a uniform *probability* and find which level it falls in. The
        # levels are deliberately not evenly spaced -- dense in the tail --
        # so drawing a uniform index instead would oversample the tail
        # badly: 60 of 460 levels sit above the 99th percentile.
        uniform = torch.rand(batch, self.n_cells, draws, device=self.table.device)
        picks = torch.searchsorted(self.levels, uniform.reshape(-1).contiguous())
        picks = picks.clamp(max=self.n_levels - 1).reshape(batch, self.n_cells, draws)

        chosen = self.table[months]  # (B, n_cells, n_levels)
        drawn = torch.gather(chosen, 2, picks)
        return drawn.reshape(batch, *grid, draws)


def months_from_nanoseconds(times_ns: torch.Tensor) -> torch.Tensor:
    """Month index 0-11 from nanoseconds since the epoch.

    Done through numpy's datetime machinery rather than by arithmetic on the
    integer, which would have to account for leap years by hand.
    """
    as_dates = times_ns.detach().cpu().numpy().astype("datetime64[ns]")
    months = as_dates.astype("datetime64[M]").astype(int) % 12
    return torch.as_tensor(months, dtype=torch.long)


class PersistenceBaseline:
    """The last observed field, carried forward.

    A degenerate distribution: one value repeated, like the point heads. Its
    exceedance statistics match observations almost exactly, because it *is*
    observations, offset by one step.
    """

    def __init__(self, channel: int) -> None:
        self.channel = channel

    def samples(self, last_step_mm: torch.Tensor, draws: int) -> torch.Tensor:
        """``(B, H, W, draws)``, every draw identical."""
        return last_step_mm.unsqueeze(-1).expand(-1, -1, -1, draws)
