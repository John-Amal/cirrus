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
from cirrus.eval.compare import load_arm
from cirrus.eval.gpd import (
    GpdFit,
    bootstrap_return_levels,
    fit_from_exceedances,
    fit_gpd,
)
from cirrus.models.patch_embed import flatten_time
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
    spec: FinetuneSpec,
    head: torch.nn.Module,
    backbone: ViT,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    to_mm: PrecipitationTarget,
    channel: int,
    thresholds: torch.Tensor,
    draws: int,
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
        x = flatten_time(batch["input"]).to(device)
        tokens = backbone(x)
        if spec.is_distributional:
            samples = head(tokens).sample(draws)
        else:
            samples = head(tokens).unsqueeze(-1).expand(-1, -1, -1, draws)

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
    grid = thresholds.shape

    store = xr.open_zarr(data_spec.output, chunks=None)
    latitudes = np.asarray(store["latitude"].values, dtype=np.float32)
    weights = np.repeat(np.cos(np.deg2rad(latitudes)), grid[1])

    print(f"observations: training {splits.train.start}..{splits.train.end}")
    train_field = observed_field(
        data_spec.output, variable, splits.train.start, splits.train.end
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
    del train_field
    print(
        f"shape stationarity: training xi inside the test bootstrap interval "
        f"for {stationarity['coverage']:.0%} of {int(stationarity['cells_compared'])} "
        f"cells (median |difference| {stationarity['median_abs_difference']:.3f})"
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
        print(f"sampling {arm} over the test years ({draws} draws per step)...")
        arm_cells[arm] = sample_arm(
            spec,
            head,
            backbone,
            loader,
            device,
            to_mm,
            channel,
            thresholds_tensor,
            draws,
            seed,
        )

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
        "variants": {},
    }

    for variant in SHAPE_VARIANTS:
        if variant == "free":
            fixed = None
        elif variant == "fixed_train":
            fixed = shapes_train
        else:
            fixed = np.full(grid, shape_global)

        observed_fits = fit_cells(test_cells, thresholds, fixed)
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
