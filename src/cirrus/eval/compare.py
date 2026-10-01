"""Comparing the objective arms on common ground.

Training losses cannot be compared across arms: CRPS and squared error are
different quantities on different scales. So every arm is re-scored here with
the same metrics, on the same data, from the same fixed masking.

Each arm is represented as **samples** — a point head simply returns its
value repeated — so the same code scores all four without special cases, and
a point prediction is treated as the degenerate distribution it is.

The metrics answer different questions on purpose:

``mae_mm``
    Bulk accuracy. The arm that optimises L1 should win this, and that is
    not the interesting question.
``crps``
    Proper score over the whole distribution. Rewards being right *and*
    honestly dispersed; a point prediction scores its absolute error.
``twcrps``
    The same, restricted above each cell's exceedance threshold.
``brier``
    Skill at the yes/no question "will this cell exceed its threshold?",
    which is what a warning system actually asks.
``p99`` / ``p999`` / ``max`` ratios
    Predicted over true quantiles, pooled across cells. The Phase 2 baseline
    was 0.30 at p99.9. This is the number the tail-aware objective exists to
    move.

Amplitude is reported two ways, because they mean different things. The
**mean field** is what you get by asking each cell for one number, and is
smooth by construction. A **draw** is one realisation from the predictive
distribution, and keeps the dispersion the model actually claims. A
deterministic arm has no draw distinct from its mean, which is precisely the
limitation being measured.
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
from cirrus.data.thresholds import Thresholds
from cirrus.device import device_report, get_device
from cirrus.eval.baselines import Climatology, ClimatologyBaseline
from cirrus.eval.samplers import (
    Sampler,
    climatology_sampler,
    persistence_sampler,
    trained_sampler,
)
from cirrus.losses.crps import crps_sample, mean_score, threshold_weighted_crps
from cirrus.models.vit import ViT
from cirrus.train.finetune import (
    FinetuneSpec,
    PrecipitationTarget,
    build_head,
    load_backbone,
)

QUANTILES = (0.99, 0.999)


@dataclass
class ArmScores:
    """Everything measured for one arm."""

    label: str
    metrics: dict[str, float]
    amplitude: dict[str, dict[str, float]]


def load_arm(
    run_dir: Path, backbone: ViT, device: torch.device
) -> tuple[FinetuneSpec, torch.nn.Module]:
    """Rebuild a trained head from its checkpoint."""
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    spec = FinetuneSpec(**state["spec"])
    head = build_head(spec.objective, backbone, spec.head_depth)
    head.load_state_dict(state["head"])
    return spec, head.to(device).eval()


@torch.no_grad()
def score_arm(
    label: str,
    sampler: Sampler,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    to_mm: PrecipitationTarget,
    channel: int,
    thresholds: torch.Tensor,
    weights: torch.Tensor,
    batches: int,
    seed: int = 0,
) -> ArmScores:
    """Score anything that can produce samples, on a fixed subset of data."""
    totals: dict[str, float] = {"mae_mm": 0.0, "crps": 0.0, "twcrps": 0.0, "brier": 0.0}
    seen = 0
    truth_parts, mean_parts, draw_parts = [], [], []

    for index, batch in enumerate(loader):
        if index >= batches:
            break
        torch.manual_seed(seed + index)  # identical draws for every arm
        target = to_mm(batch["target"][:, 0, channel].to(device))
        samples = sampler(batch)

        fair = samples.shape[-1] > 1  # the fair estimator needs two samples
        predicted_mean = samples.mean(dim=-1)
        exceedance = (samples > thresholds.unsqueeze(-1)).to(samples.dtype).mean(-1)
        observed = (target > thresholds).to(samples.dtype)

        totals["mae_mm"] += float(mean_score((predicted_mean - target).abs(), weights))
        totals["crps"] += float(mean_score(crps_sample(samples, target, fair), weights))
        totals["twcrps"] += float(
            mean_score(
                threshold_weighted_crps(samples, target, thresholds, fair), weights
            )
        )
        totals["brier"] += float(mean_score((exceedance - observed) ** 2, weights))
        seen += 1

        truth_parts.append(target.cpu().numpy().ravel())
        mean_parts.append(predicted_mean.cpu().numpy().ravel())
        draw_parts.append(samples[..., 0].cpu().numpy().ravel())

    metrics = {name: value / max(seen, 1) for name, value in totals.items()}
    truth = np.concatenate(truth_parts)
    amplitude = {
        "mean_field": quantile_ratios(np.concatenate(mean_parts), truth),
        "single_draw": quantile_ratios(np.concatenate(draw_parts), truth),
    }
    return ArmScores(label, metrics, amplitude)


def quantile_ratios(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Predicted over true quantiles, pooled over all cells and times."""
    ratios = {"mean": float(predicted.mean() / truth.mean())}
    for q in QUANTILES:
        true_value = float(np.quantile(truth, q))
        ratios[f"p{q * 100:g}"] = (
            float(np.quantile(predicted, q)) / true_value
            if true_value
            else float("nan")
        )
    ratios["max"] = float(predicted.max() / truth.max())
    return ratios


def compare(
    run_root: str | Path = "runs",
    split: str = "val",
    batches: int = 40,
    data_config: str | Path = "configs/data/era5_5625.yaml",
    normalise_config: str | Path = "configs/data/normalise.yaml",
    thresholds_path: str | Path = "data/stats/thresholds_train.json",
    variable: str = "total_precipitation_6hr",
    out_path: str | Path = "runs/comparison.json",
    climatology_path: str | Path = "data/stats/climatology_train.npz",
    workers: int = 4,
) -> list[ArmScores]:
    """Score every trained arm and print the comparison table."""
    device = get_device()
    print(f"device: {device_report(device)}   split: {split}")

    data_spec = IngestSpec.from_yaml(data_config)
    channel = data_spec.time_channels.index(variable)
    normaliser = Normaliser.load(NormaliseSpec.from_yaml(normalise_config).output)
    to_mm = PrecipitationTarget(normaliser, variable)

    dataset = ERA5Dataset.from_configs(split=split, data=data_config)
    loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )

    store = xr.open_zarr(data_spec.output, chunks=None)
    latitudes = torch.as_tensor(store["latitude"].values.copy())
    weights = torch.cos(torch.deg2rad(latitudes.float()))
    weights = (weights / weights.mean())[:, None].to(device)
    thresholds = torch.as_tensor(
        Thresholds.load(thresholds_path).values, dtype=torch.float32
    ).to(device)

    # Discover arms rather than hard-coding them, so a variant run (a
    # different threshold, a different seed) appears without a code change.
    run_dirs = sorted(
        d for d in Path(run_root).glob("finetune_*") if (d / "best.pt").exists()
    )
    if not run_dirs:
        raise ValueError(f"no finetuned arms found under {run_root}")

    results: list[ArmScores] = []

    # Baselines first, so every table starts with the reference points.
    grid: tuple[int, int] = (int(thresholds.shape[0]), int(thresholds.shape[1]))
    samplers: list[tuple[str, Sampler]] = [
        ("persistence", persistence_sampler(to_mm, channel, device))
    ]
    climatology_file = Path(climatology_path)
    if climatology_file.exists():
        baseline = ClimatologyBaseline(Climatology.load(climatology_file), device)
        samplers.append(("climatology", climatology_sampler(baseline, grid, device)))
    else:
        print(f"no climatology at {climatology_file}; run 'cirrus climatology'")

    backbone: ViT | None = None
    for run_dir in run_dirs:
        arm = run_dir.name.removeprefix("finetune_")
        if backbone is None:
            state = torch.load(
                run_dir / "best.pt", map_location="cpu", weights_only=False
            )
            pretrained = FinetuneSpec(**state["spec"]).pretrained
            backbone, _ = load_backbone(pretrained, freeze=True)
            backbone = backbone.to(device).eval()

        spec, head = load_arm(run_dir, backbone, device)
        samplers.append((arm, trained_sampler(spec, head, backbone, device)))

    for label, sampler in samplers:
        print(f"scoring {label}...")
        results.append(
            score_arm(
                label,
                sampler,
                loader,
                device,
                to_mm,
                channel,
                thresholds,
                weights,
                batches,
            )
        )

    report(results)
    payload: list[dict[str, Any]] = [
        {"arm": r.label, "metrics": r.metrics, "amplitude": r.amplitude}
        for r in results
    ]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2))
    print(f"\nwritten: {out_path}")
    return results


def report(results: list[ArmScores]) -> None:
    """Print the comparison tables."""
    if not results:
        print("no arms to compare")
        return

    width = max(10, max(len(r.label) for r in results) + 1)
    print(
        f"\n{'arm':{width}s} {'MAE mm':>9s} {'CRPS':>9s} {'twCRPS':>9s} {'Brier':>9s}"
    )
    for row in results:
        m = row.metrics
        print(
            f"{row.label:{width}s} {m['mae_mm']:9.4f} {m['crps']:9.4f} "
            f"{m['twcrps']:9.4f} {m['brier']:9.4f}"
        )

    for space in ("mean_field", "single_draw"):
        print(f"\namplitude ratios, {space.replace('_', ' ')} (1.00 = right)")
        print(f"{'arm':{width}s} {'mean':>9s} {'p99':>9s} {'p99.9':>9s} {'max':>9s}")
        for row in results:
            a = row.amplitude[space]
            print(
                f"{row.label:{width}s} {a['mean']:9.2f} {a['p99']:9.2f} "
                f"{a['p99.9']:9.2f} {a['max']:9.2f}"
            )
