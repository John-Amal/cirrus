"""Return levels on the held-out test years.

This is the analysis the project points at, and the only one that touches
2017-2022. Everything before it was chosen on validation data.

The comparison is deliberately symmetric. Each arm's predictive
distributions are pooled over the test period to give its **implied
climatology**; a generalised Pareto is fitted to exceedances of the per-cell
threshold; return levels follow. Exactly the same procedure is applied to the
observations. Any difference is then the model's tail, not the estimator's.

The shape parameter is handled three ways, because no single choice is
defensible on its own:

``free``
    ``xi`` estimated per cell, separately for the model and for
    observations. Noisy, but it is the only variant that can detect a model
    whose tail *shape* is wrong rather than its scale.
``fixed_train``
    ``xi`` estimated per cell from the long training record and applied to
    both sides. Lower variance, and an error in ``xi`` largely cancels in the
    model/observation ratio. Primary.
``fixed_global``
    A single ``xi`` everywhere, from all training exceedances pooled. The
    crudest, and a useful check that conclusions do not rest on the
    per-cell estimates.

If the ordering of arms is the same in all three, it is robust to the
choice. If it is not, that is the more important finding.

Memory note: a GPD fit needs only the exceedances and the total count, so
model samples are streamed and only the ~2% above threshold are kept. The
alternative -- materialising an implied climatology of every sample at every
cell and time -- would be tens of gigabytes for nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

from cirrus.data.dataset import ERA5Dataset
from cirrus.data.ingest import IngestSpec
from cirrus.data.normalise import Normaliser, NormaliseSpec
from cirrus.data.splits import Splits
from cirrus.data.thresholds import Thresholds
from cirrus.device import device_report, get_device
from cirrus.eval.baselines import Climatology, ClimatologyBaseline
from cirrus.eval.compare import load_arm
from cirrus.eval.gpd import (
    GpdFit,
    bootstrap_return_levels,
    fit_from_exceedances,
    fit_gpd,
)
from cirrus.eval.samplers import (
    Sampler,
    climatology_sampler,
    persistence_sampler,
    trained_sampler,
)
from cirrus.models.vit import ViT
from cirrus.train.finetune import FinetuneSpec, PrecipitationTarget, load_backbone

RETURN_YEARS = (1.0, 5.0, 20.0)
SHAPE_VARIANTS = ("free", "fixed_train", "fixed_global")
BLOCK_SIZE = 120  # 30 days of 6-hourly steps, for the observation bootstrap


@dataclass
class CellSample:
    """Exceedances of one cell, plus how many values they came from."""

    exceedances: np.ndarray
    n_total: int

    @property
    def rate(self) -> float:
        """Fraction of values above the threshold."""
        return len(self.exceedances) / max(self.n_total, 1)


def observed_field(
    store: str | Path, variable: str, start: str, end: str, units_scale: float = 1000.0
) -> np.ndarray:
    """Observed precipitation over a period, ``(time, lat, lon)`` in mm."""
    data = xr.open_zarr(store, chunks=None)[variable].sel(time=slice(start, end))
    values: np.ndarray = np.clip(data.values.astype(np.float32), 0.0, None)
    return values * units_scale


@torch.no_grad()
def sample_arm(
    sampler: Sampler,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    thresholds: torch.Tensor,
    seed: int = 0,
) -> list[CellSample]:
    """Stream the test period, keeping only exceedances per cell.

    Grouping by cell happens once at the end, with a single sort, rather than
    per batch. The per-batch version -- a Python loop over up to 2048 cells,
    274 times -- was the difference between minutes and hours.

    Returns one entry per grid cell, flattened row-major.
    """
    n_cells = int(thresholds.numel())
    flat_thresholds = thresholds.reshape(-1)
    cell_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    n_total = 0

    for index, batch in enumerate(loader):
        torch.manual_seed(seed + index)
        samples = sampler(batch)
        draws = samples.shape[-1]
        flat = samples.reshape(samples.shape[0], n_cells, draws)
        above = flat > flat_thresholds[None, :, None]
        n_total += flat.shape[0] * draws

        positions = torch.nonzero(above, as_tuple=False)
        if positions.numel():
            cell_parts.append(positions[:, 1].cpu().numpy())
            value_parts.append(flat[above].cpu().numpy())

    empty = np.empty(0, dtype=np.float32)
    if not cell_parts:
        return [CellSample(empty, n_total) for _ in range(n_cells)]

    cells = np.concatenate(cell_parts)
    values = np.concatenate(value_parts).astype(np.float32)

    # One stable sort, then split on the per-cell counts: O(N log N) once
    # instead of a mask over every cell for every batch.
    order = np.argsort(cells, kind="stable")
    sorted_values = values[order]
    counts = np.bincount(cells, minlength=n_cells)
    boundaries = np.cumsum(counts)[:-1]
    grouped = np.split(sorted_values, boundaries)
    return [CellSample(group, n_total) for group in grouped]


def fit_cells(
    samples: list[CellSample],
    thresholds: np.ndarray,
    fixed_shapes: np.ndarray | None,
) -> list[GpdFit]:
    """Fit every cell from pre-extracted exceedances."""
    flat_thresholds = thresholds.reshape(-1)
    fits: list[GpdFit] = []
    for index, sample in enumerate(samples):
        fixed = None
        if fixed_shapes is not None:
            candidate = float(fixed_shapes.reshape(-1)[index])
            fixed = candidate if np.isfinite(candidate) else None
        fits.append(
            fit_from_exceedances(
                sample.exceedances,
                float(flat_thresholds[index]),
                sample.n_total,
                fixed,
            )
        )
    return fits


def field_to_cells(values: np.ndarray, thresholds: np.ndarray) -> list[CellSample]:
    """Turn a ``(time, lat, lon)`` field into per-cell exceedance samples."""
    n_time = values.shape[0]
    flat = values.reshape(n_time, -1)
    flat_thresholds = thresholds.reshape(-1)
    return [
        CellSample(
            flat[:, cell][flat[:, cell] > flat_thresholds[cell]].astype(np.float32),
            n_time,
        )
        for cell in range(flat.shape[1])
    ]


def shape_field(fits: list[GpdFit], grid: tuple[int, int]) -> np.ndarray:
    """Per-cell shape parameters as a grid, NaN where the fit is invalid."""
    values = np.array([fit.shape if fit.is_valid else np.nan for fit in fits])
    return values.reshape(grid)


def return_levels(fits: list[GpdFit], years: float) -> np.ndarray:
    """Return levels for every cell, NaN where the fit is invalid."""
    return np.array([fit.return_level(years) for fit in fits])


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Latitude-weighted median, ignoring NaNs."""
    usable = np.isfinite(values)
    if not usable.any():
        return float("nan")
    order = np.argsort(values[usable])
    sorted_values = values[usable][order]
    sorted_weights = weights[usable][order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = 0.5 * cumulative[-1]
    return float(sorted_values[int(np.searchsorted(cumulative, cutoff))])


def bootstrap_ratio(
    ratios: np.ndarray, weights: np.ndarray, n_resamples: int = 500, seed: int = 0
) -> tuple[float, float]:
    """Interval on the weighted-median ratio, resampling grid cells.

    This is *spatial* uncertainty: how much the headline number depends on
    which cells happened to be included. It is not temporal uncertainty,
    which is what the block bootstrap below measures for individual cells.
    Both are real and they are not interchangeable.
    """
    usable = np.isfinite(ratios)
    values, cell_weights = ratios[usable], weights[usable]
    if values.size < 10:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples)
    for index in range(n_resamples):
        picks = rng.integers(0, values.size, size=values.size)
        estimates[index] = weighted_median(values[picks], cell_weights[picks])
    return float(np.percentile(estimates, 5)), float(np.percentile(estimates, 95))


def observed_uncertainty(
    test_field: np.ndarray,
    thresholds: np.ndarray,
    years: float,
    n_cells: int = 24,
    seed: int = 0,
) -> float:
    """Typical width of a block-bootstrap interval on an observed level.

    Reported as a fraction of the point estimate, over a random subset of
    cells. Exceedances cluster in time, so the blocks matter: an ordinary
    bootstrap would report intervals that are too narrow, and the comparison
    would look more decisive than the data supports.
    """
    rng = np.random.default_rng(seed)
    flat = test_field.reshape(test_field.shape[0], -1)
    flat_thresholds = thresholds.reshape(-1)
    widths: list[float] = []

    for cell in rng.permutation(flat_thresholds.size)[: n_cells * 3]:
        if len(widths) >= n_cells:
            break
        low, point, high = bootstrap_return_levels(
            flat[:, cell],
            float(flat_thresholds[cell]),
            years,
            n_resamples=120,
            block_size=BLOCK_SIZE,
            seed=int(cell),
        )
        if np.isfinite(low) and np.isfinite(high) and point > 0:
            widths.append((high - low) / point)
    return float(np.median(widths)) if widths else float("nan")


def period_comparison(
    train_fits: list[GpdFit],
    test_fits: list[GpdFit],
    weights: np.ndarray,
) -> dict[str, float]:
    """Observed return levels in the test period relative to the training one.

    Both periods go through the identical fit, so this isolates whether the
    tail itself differs between 1979-2014 and 2017-2022. It matters for
    reading the climatology baseline: a baseline built from training years
    and scored on test years will under-predict if the test period is
    genuinely heavier, and that is a property of the climate rather than a
    defect of the baseline.

    Return levels are rate-aware, so the unequal record lengths do not bias
    the comparison.
    """
    ratios: dict[str, float] = {}
    for years in RETURN_YEARS:
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = return_levels(test_fits, years) / return_levels(train_fits, years)
        ratios[f"{years:g}y"] = weighted_median(ratio, weights)
    return ratios


def truncation_check(
    climatology_path: str | Path,
    test_fits: list[GpdFit],
    years: float = 20.0,
) -> float:
    """Share of cells where the climatology table cannot reach the test level.

    The stored quantiles stop at the top level, so a climatological sample
    can never exceed roughly the wettest value in the training record. Where
    that ceiling sits below the observed return level, the baseline's
    shortfall is a resolution artifact rather than a statement about the
    climate. This separates the two explanations.
    """
    path = Path(climatology_path)
    if not path.exists():
        return float("nan")

    climatology = Climatology.load(path)
    ceiling = climatology.quantiles[:, :, -1].max(axis=0)  # best case over months
    observed = return_levels(test_fits, years)
    usable = np.isfinite(observed)
    return float((ceiling[usable] < observed[usable]).mean())


def block_variability(
    train_field: np.ndarray,
    train_times: np.ndarray,
    thresholds: np.ndarray,
    weights: np.ndarray,
    reference_fits: list[GpdFit],
    fixed_shapes: np.ndarray,
    block_years: int = 6,
    years: float = 20.0,
) -> dict[str, Any]:
    """How much a short window's tail wanders, from internal variability alone.

    The training record is cut into consecutive windows the same length as
    the test period, and each is compared against the full-record fit. None
    of them can differ from that reference for a *forced* reason, so the
    spread is what interannual variability alone produces in a window this
    short.

    A test-period ratio inside that spread is not evidence of a trend. One
    outside it is, and a monotone drift across the blocks themselves would be
    a trend visible within the training data.
    """
    years_available = train_times.astype("datetime64[Y]").astype(int) + 1970
    first, last = int(years_available.min()), int(years_available.max())
    reference = return_levels(reference_fits, years)

    blocks: list[dict[str, float]] = []
    for start in range(first, last - block_years + 2, block_years):
        rows = (years_available >= start) & (years_available < start + block_years)
        if rows.sum() < 1000:
            continue
        cells = field_to_cells(train_field[rows], thresholds)
        fits = fit_cells(cells, thresholds, fixed_shapes)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = return_levels(fits, years) / reference
        blocks.append({"start": float(start), "ratio": weighted_median(ratio, weights)})

    values = np.array([block["ratio"] for block in blocks])
    return {
        "blocks": blocks,
        "min": float(values.min()) if values.size else float("nan"),
        "max": float(values.max()) if values.size else float("nan"),
        "spread": float(values.max() - values.min()) if values.size else float("nan"),
    }


def shape_stationarity(
    train_values: np.ndarray,
    test_values: np.ndarray,
    thresholds: np.ndarray,
    n_cells: int = 40,
    seed: int = 0,
) -> dict[str, float]:
    """Compare the tail shape in 1979-2014 against 2017-2022.

    Fits ``xi`` independently on each period for a random subset of cells and
    asks how often the training estimate falls inside a bootstrap interval
    for the test estimate. High coverage means the stationarity assumption
    behind ``fixed_train`` is consistent with the data; low coverage means it
    is not, and the ``free`` variant should be preferred.
    """
    rng = np.random.default_rng(seed)
    flat_train = train_values.reshape(train_values.shape[0], -1)
    flat_test = test_values.reshape(test_values.shape[0], -1)
    flat_thresholds = thresholds.reshape(-1)

    candidates = rng.permutation(flat_thresholds.size)[: n_cells * 3]
    inside, compared, differences = 0, 0, []

    for cell in candidates:
        if compared >= n_cells:
            break
        threshold = float(flat_thresholds[cell])
        train_fit = fit_gpd(flat_train[:, cell], threshold)
        test_fit = fit_gpd(flat_test[:, cell], threshold)
        if not (train_fit.is_valid and test_fit.is_valid):
            continue

        resampled = []
        exceedances = flat_test[:, cell][flat_test[:, cell] > threshold]
        for _ in range(120):
            draw = rng.choice(exceedances, size=len(exceedances), replace=True)
            resampled.append(
                fit_from_exceedances(draw, threshold, flat_test.shape[0]).shape
            )
        low, high = np.percentile(resampled, [5, 95])

        inside += int(low <= train_fit.shape <= high)
        differences.append(abs(train_fit.shape - test_fit.shape))
        compared += 1

    return {
        "cells_compared": float(compared),
        "coverage": inside / max(compared, 1),
        "median_abs_difference": float(np.median(differences))
        if differences
        else float("nan"),
    }


def global_shape(cells: list[CellSample], thresholds: np.ndarray) -> float:
    """One shape parameter from every training exceedance pooled.

    Excesses are pooled rather than raw values, since each cell has its own
    threshold. The crudest of the three variants, and the least able to hide
    a problem behind per-cell noise.
    """
    flat_thresholds = thresholds.reshape(-1)
    excesses = np.concatenate(
        [
            sample.exceedances - flat_thresholds[index]
            for index, sample in enumerate(cells)
            if len(sample.exceedances)
        ]
    )
    pooled = fit_from_exceedances(excesses, 0.0, len(excesses))
    return pooled.shape


def evaluate_return_levels(
    run_root: str | Path = "runs",
    draws: int = 4,
    data_config: str | Path = "configs/data/era5_5625.yaml",
    normalise_config: str | Path = "configs/data/normalise.yaml",
    splits_config: str | Path = "configs/data/splits.yaml",
    thresholds_path: str | Path = "data/stats/thresholds_train.json",
    variable: str = "total_precipitation_6hr",
    out_path: str | Path = "runs/returnlevels.json",
    climatology_path: str | Path = "data/stats/climatology_train.npz",
    stationarity_cells: int = 40,
    workers: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """Return levels for every arm on the test years, three shape variants."""
    device = get_device()
    print(f"device: {device_report(device)}")

    data_spec = IngestSpec.from_yaml(data_config)
    splits = Splits.from_yaml(splits_config)
    channel = data_spec.time_channels.index(variable)
    normaliser = Normaliser.load(NormaliseSpec.from_yaml(normalise_config).output)
    to_mm = PrecipitationTarget(normaliser, variable)
    thresholds_obj = Thresholds.load(thresholds_path)
    thresholds = thresholds_obj.values
    grid: tuple[int, int] = (int(thresholds.shape[0]), int(thresholds.shape[1]))

    store = xr.open_zarr(data_spec.output, chunks=None)
    latitudes = np.asarray(store["latitude"].values, dtype=np.float32)
    weights = np.repeat(np.cos(np.deg2rad(latitudes)), grid[1])

    print(f"observations: training {splits.train.start}..{splits.train.end}")
    train_field = observed_field(
        data_spec.output, variable, splits.train.start, splits.train.end
    )
    train_times = (
        xr.open_zarr(data_spec.output, chunks=None)["time"]
        .sel(time=slice(splits.train.start, splits.train.end))
        .values
    )
    train_cells = field_to_cells(train_field, thresholds)
    train_fits = fit_cells(train_cells, thresholds, None)
    shapes_train = shape_field(train_fits, grid)
    shape_global = global_shape(train_cells, thresholds)
    print(
        f"training shape: median {np.nanmedian(shapes_train):+.3f}, "
        f"pooled global {shape_global:+.3f}"
    )

    print(f"observations: test {splits.test.start}..{splits.test.end}")
    test_field = observed_field(
        data_spec.output, variable, splits.test.start, splits.test.end
    )
    test_cells = field_to_cells(test_field, thresholds)

    stationarity = shape_stationarity(
        train_field, test_field, thresholds, stationarity_cells, seed
    )
    variability = block_variability(
        train_field,
        train_times,
        thresholds,
        weights,
        fit_cells(train_cells, thresholds, shapes_train),
        shapes_train,
    )
    del train_field
    print(
        f"shape stationarity: training xi inside the test bootstrap interval "
        f"for {stationarity['coverage']:.0%} of {int(stationarity['cells_compared'])} "
        f"cells (median |difference| {stationarity['median_abs_difference']:.3f})"
    )

    # Fit observations once per shape variant and reuse. Refitting 2,048
    # cells is seconds of maximum likelihood each time, and the naive
    # arrangement did it nine times over.
    fixed_by_variant: dict[str, np.ndarray | None] = {
        "free": None,
        "fixed_train": shapes_train,
        "fixed_global": np.full(grid, shape_global),
    }
    observed_fits_by_variant = {
        name: fit_cells(test_cells, thresholds, fixed)
        for name, fixed in fixed_by_variant.items()
    }
    train_fits_fixed = fit_cells(train_cells, thresholds, shapes_train)

    trend_free = period_comparison(
        train_fits, observed_fits_by_variant["free"], weights
    )
    trend_fixed = period_comparison(
        train_fits_fixed, observed_fits_by_variant["fixed_train"], weights
    )
    truncated = truncation_check(climatology_path, observed_fits_by_variant["free"])
    print(
        "six-year windows within training vary over "
        f"{variability['min']:.2f}–{variability['max']:.2f} "
        f"({len(variability['blocks'])} windows); anything inside that range "
        "is internal variability"
    )
    print(
        "observed test/training return level: "
        + ", ".join(f"{k} {v:.2f}" for k, v in trend_free.items())
        + " (free shape), "
        + ", ".join(f"{k} {v:.2f}" for k, v in trend_fixed.items())
        + " (fixed shape)"
    )
    print(
        f"climatology table ceiling below the observed 20y level in "
        f"{truncated:.1%} of cells"
    )

    run_dirs = sorted(
        d for d in Path(run_root).glob("finetune_*") if (d / "best.pt").exists()
    )
    if not run_dirs:
        raise ValueError(f"no finetuned arms under {run_root}")

    dataset = ERA5Dataset.from_configs(split="test", data=data_config)
    # Workers matter more here than anywhere else: this reads the whole test
    # period once per arm, and single-process loading from zarr is ~4.6 s per
    # batch against ~0.3 s of compute. Without them the run is loader-bound
    # by more than an order of magnitude.
    loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    thresholds_tensor = torch.as_tensor(thresholds, dtype=torch.float32).to(device)

    samplers: list[tuple[str, Sampler]] = [
        ("persistence", persistence_sampler(to_mm, channel, device))
    ]
    climatology_file = Path(climatology_path)
    if climatology_file.exists():
        baseline = ClimatologyBaseline(Climatology.load(climatology_file), device)
        samplers.append(
            ("climatology", climatology_sampler(baseline, grid, device, draws))
        )
    else:
        print(f"no climatology at {climatology_file}; run 'cirrus climatology'")

    arm_cells: dict[str, list[CellSample]] = {}
    backbone: ViT | None = None
    for run_dir in run_dirs:
        arm = run_dir.name.removeprefix("finetune_")
        if backbone is None:
            state = torch.load(
                run_dir / "best.pt", map_location="cpu", weights_only=False
            )
            backbone, _ = load_backbone(
                FinetuneSpec(**state["spec"]).pretrained, freeze=True
            )
            backbone = backbone.to(device).eval()
        spec, head = load_arm(run_dir, backbone, device)
        samplers.append((arm, trained_sampler(spec, head, backbone, device, draws)))

    for label, sampler in samplers:
        print(f"sampling {label} over the test years ({draws} draws per step)...")
        arm_cells[label] = sample_arm(sampler, loader, device, thresholds_tensor, seed)

    observed_widths = {
        f"{years:g}y": observed_uncertainty(test_field, thresholds, years, seed=seed)
        for years in RETURN_YEARS
    }
    print(
        "observed return levels carry a block-bootstrap interval of "
        + ", ".join(f"{k} +/-{v / 2:.0%}" for k, v in observed_widths.items())
    )

    results: dict[str, Any] = {
        "meta": {
            "test_period": [splits.test.start, splits.test.end],
            "observed_interval_width": observed_widths,
            "draws_per_step": draws,
            "threshold_rate": thresholds_obj.meta["exceedance_rate"],
            "shape_train_median": float(np.nanmedian(shapes_train)),
            "shape_global": shape_global,
        },
        "stationarity": stationarity,
        "internal_variability": variability,
        "period_comparison": {"free": trend_free, "fixed_train": trend_fixed},
        "climatology_truncated_fraction": truncated,
        "variants": {},
    }

    for variant in SHAPE_VARIANTS:
        fixed = fixed_by_variant[variant]
        observed_fits = observed_fits_by_variant[variant]
        entry: dict[str, Any] = {"arms": {}}
        for arm, cells in arm_cells.items():
            model_fits = fit_cells(cells, thresholds, fixed)
            ratios: dict[str, float] = {}
            intervals: dict[str, list[float]] = {}
            for years in RETURN_YEARS:
                model_levels = return_levels(model_fits, years)
                observed_levels = return_levels(observed_fits, years)
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = model_levels / observed_levels
                ratios[f"{years:g}y"] = weighted_median(ratio, weights)
                low, high = bootstrap_ratio(ratio, weights, seed=seed)
                intervals[f"{years:g}y"] = [low, high]
            entry["arms"][arm] = {
                "return_level_ratio": ratios,
                "ratio_interval_spatial": intervals,
                "median_shape": float(
                    np.nanmedian([f.shape for f in model_fits if f.is_valid])
                ),
                "valid_cells": int(sum(f.is_valid for f in model_fits)),
            }
        entry["observed_median_shape"] = float(
            np.nanmedian([f.shape for f in observed_fits if f.is_valid])
        )
        results["variants"][variant] = entry

    report_return_levels(results)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2))
    print(f"\nwritten: {out_path}")
    return results


def report_return_levels(results: dict[str, Any]) -> None:
    """Print one table per shape variant."""
    for variant, entry in results["variants"].items():
        arms = entry["arms"]
        width = max(10, max(len(a) for a in arms) + 1)
        print(f"\nreturn level ratio, model / observed — shape: {variant}")
        print(f"(observed median xi {entry['observed_median_shape']:+.3f})")
        header = "  ".join(
            f"{y:>8s}" for y in next(iter(arms.values()))["return_level_ratio"]
        )
        print(f"{'arm':{width}s} {header}  {'xi':>8s}")
        for arm, values in arms.items():
            row = "  ".join(f"{v:8.2f}" for v in values["return_level_ratio"].values())
            print(f"{arm:{width}s} {row}  {values['median_shape']:+8.3f}")
