"""Fine-tuning precipitation heads on the pretrained backbone.

This is the experiment. Four objectives, one frozen backbone, one dataset,
one schedule -- so any difference in the results is attributable to the
objective and nothing else:

===========  =========================================================
``mse``      point prediction, squared error. Standard practice.
``l1``       point prediction, absolute error. Less tail-averse than MSE.
``crps``     CSGD distribution, continuous ranked probability score.
``twcrps``   CSGD distribution, scored above the per-cell threshold.
===========  =========================================================

Three choices keep the comparison fair.

**The backbone is frozen.** Every arm starts from identical representations,
so the heads are the only thing that differ. It also makes each run minutes
rather than hours, since the encoder runs without gradients.

**Targets are in millimetres.** Phase 2 measured a 24% dry bias created by
training in ``log1p`` space and transforming back. Predicting physical units
removes that mechanism from the comparison entirely.

**Validation reports mean absolute error in millimetres for every arm**,
alongside each arm's own objective. Training losses are not comparable across
arms -- CRPS and MSE are different quantities -- so model selection uses the
arm's own score, and the cross-arm table uses MAE. For a point prediction,
CRPS reduces exactly to absolute error, which is what makes the comparison
coherent at all.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import xarray as xr
from torch import nn
from torch.utils.data import DataLoader

from cirrus.config import read_yaml_mapping
from cirrus.data.dataset import ERA5Dataset
from cirrus.data.ingest import IngestSpec
from cirrus.data.normalise import Normaliser, NormaliseSpec
from cirrus.data.thresholds import Thresholds
from cirrus.device import device_report, get_device
from cirrus.losses.crps import crps_sample, mean_score, threshold_weighted_crps
from cirrus.models.heads.csgd import CsgdHead, PointHead
from cirrus.models.patch_embed import flatten_time
from cirrus.models.vit import BackboneSpec, ViT
from cirrus.train.pretrain import learning_rate_at, parameter_groups

OBJECTIVES = ("mse", "l1", "crps", "twcrps")


@dataclass(frozen=True)
class FinetuneSpec:
    """One arm of the objective comparison."""

    objective: str = "twcrps"
    pretrained: str = "runs/mae_small/best.pt"
    freeze_backbone: bool = True
    head_depth: int = 2
    n_samples: int = 24  # CRPS spread term is O(n^2); 20-30 is the useful range

    epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_epochs: float = 0.5
    min_lr_ratio: float = 0.01
    grad_clip: float = 1.0
    num_workers: int = 4
    seed: int = 0
    log_every: int = 50
    max_steps_per_epoch: int = 0
    val_batches: int = 20
    out_dir: str = "runs"

    def __post_init__(self) -> None:
        """Reject an unknown objective before anything is loaded."""
        if self.objective not in OBJECTIVES:
            raise ValueError(
                f"objective must be one of {OBJECTIVES}, got {self.objective!r}"
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> FinetuneSpec:
        """Load a spec, rejecting unknown keys."""
        raw: dict[str, Any] = read_yaml_mapping(path, (f.name for f in fields(cls)))
        return cls(**raw)

    @property
    def is_distributional(self) -> bool:
        """Whether this arm predicts a distribution rather than a value."""
        return self.objective in ("crps", "twcrps")

    @property
    def run_name(self) -> str:
        """Directory name for this arm."""
        return f"finetune_{self.objective}"


class PrecipitationTarget:
    """Convert a normalised precipitation channel back to millimetres.

    The dataset hands out normalised values; the heads predict millimetres.
    This is the bridge, in torch so it stays on the device and inside the
    graph if it ever needs to be.
    """

    def __init__(
        self, normaliser: Normaliser, variable: str, units_scale: float = 1000.0
    ):
        single = normaliser.subset([variable])
        self.mean = float(single.mean[0])
        self.std = float(single.std[0])
        self.is_log = variable in single.log_channels
        self.log_scale = single.log_scale
        self.units_scale = units_scale

    def __call__(self, normalised: torch.Tensor) -> torch.Tensor:
        """Normalised values to millimetres, never negative."""
        values = normalised * self.std + self.mean
        if self.is_log:
            values = torch.expm1(values) * self.log_scale
        return torch.clamp(values * self.units_scale, min=0.0)


def build_head(objective: str, backbone: ViT, head_depth: int) -> PointHead | CsgdHead:
    """Build the head for an objective: a point estimate or a distribution."""
    arguments = (backbone.spec.dim, backbone.spec.patch, backbone.grid)
    if objective in ("mse", "l1"):
        return PointHead(*arguments, depth=head_depth)
    return CsgdHead(*arguments, depth=head_depth)


def load_backbone(checkpoint: str | Path, freeze: bool) -> tuple[ViT, dict[str, Any]]:
    """Rebuild the pretrained encoder and optionally freeze it."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = {
        key[len("backbone.") :]: value
        for key, value in state["model"].items()
        if key.startswith("backbone.")
    }
    embed_weight = weights["patch_embed.proj.weight"]
    backbone = ViT(
        in_channels=int(embed_weight.shape[1]),
        spec=BackboneSpec(**state["backbone_spec"]),
        grid=(32, 64),
    )
    backbone.load_state_dict(weights)
    if freeze:
        backbone.eval()
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
    return backbone, state


def score_batch(
    spec: FinetuneSpec,
    head: nn.Module,
    tokens: torch.Tensor,
    target_mm: torch.Tensor,
    thresholds: torch.Tensor | None,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(objective, samples)`` for one batch.

    ``samples`` is ``(B, H, W, m)`` for every arm: a point head returns its
    value repeated, so downstream code -- and the MAE metric -- treats all
    four arms identically.
    """
    if spec.is_distributional:
        distribution = head(tokens)
        samples = distribution.sample(spec.n_samples)
        if spec.objective == "crps":
            per_cell = crps_sample(samples, target_mm)
        else:
            assert thresholds is not None
            per_cell = threshold_weighted_crps(samples, target_mm, thresholds)
    else:
        prediction = head(tokens)
        samples = prediction.unsqueeze(-1)
        error = prediction - target_mm
        per_cell = error**2 if spec.objective == "mse" else error.abs()
    return mean_score(per_cell, weights), samples


@torch.no_grad()
def validate(
    spec: FinetuneSpec,
    backbone: ViT,
    head: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    to_mm: PrecipitationTarget,
    channel: int,
    thresholds: torch.Tensor | None,
    weights: torch.Tensor,
) -> tuple[float, float]:
    """Return ``(objective, mae_mm)`` on a fixed subset of validation data."""
    head.eval()
    objective_total, mae_total, seen = 0.0, 0.0, 0
    for index, batch in enumerate(loader):
        if index >= spec.val_batches:
            break
        torch.manual_seed(spec.seed + index)  # same draws every epoch
        x = flatten_time(batch["input"]).to(device)
        target_mm = to_mm(batch["target"][:, 0, channel].to(device))
        tokens = backbone(x)
        objective, samples = score_batch(
            spec, head, tokens, target_mm, thresholds, weights
        )
        mae = mean_score((samples.mean(dim=-1) - target_mm).abs(), weights)
        objective_total += float(objective)
        mae_total += float(mae)
        seen += 1
    head.train()
    return objective_total / max(seen, 1), mae_total / max(seen, 1)


def finetune(
    spec: FinetuneSpec,
    data_config: str | Path = "configs/data/era5_5625.yaml",
    normalise_config: str | Path = "configs/data/normalise.yaml",
    thresholds_path: str | Path = "data/stats/thresholds_train.json",
    variable: str = "total_precipitation_6hr",
) -> Path:
    """Train one arm and return its run directory."""
    torch.manual_seed(spec.seed)
    device = get_device()
    print(f"device: {device_report(device)}   objective: {spec.objective}")

    data_spec = IngestSpec.from_yaml(data_config)
    channel = data_spec.time_channels.index(variable)
    normaliser = Normaliser.load(NormaliseSpec.from_yaml(normalise_config).output)
    to_mm = PrecipitationTarget(normaliser, variable)

    train_set = ERA5Dataset.from_configs(split="train", data=data_config)
    val_set = ERA5Dataset.from_configs(split="val", data=data_config)
    worker_options: dict[str, Any] = (
        {"persistent_workers": True, "prefetch_factor": 4}
        if spec.num_workers > 0
        else {}
    )
    train_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        train_set,
        batch_size=spec.batch_size,
        shuffle=True,
        num_workers=spec.num_workers,
        drop_last=True,
        **worker_options,
    )
    val_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        val_set,
        batch_size=spec.batch_size,
        shuffle=False,
        num_workers=spec.num_workers,
        **worker_options,
    )

    backbone, _ = load_backbone(spec.pretrained, spec.freeze_backbone)
    backbone = backbone.to(device)
    head = build_head(spec.objective, backbone, spec.head_depth).to(device)
    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(
        f"head: {trainable:,} trainable parameters, "
        f"backbone frozen={spec.freeze_backbone}"
    )

    store = xr.open_zarr(data_spec.output, chunks=None)
    latitudes = torch.as_tensor(store["latitude"].values.copy())
    weights = torch.cos(torch.deg2rad(latitudes.float()))
    weights = (weights / weights.mean())[:, None].to(device)

    thresholds = None
    if spec.objective == "twcrps":
        loaded = Thresholds.load(thresholds_path)
        thresholds = torch.as_tensor(loaded.values, dtype=torch.float32).to(device)
        print(
            f"thresholds: median {float(thresholds.median()):.2f} mm "
            f"({loaded.meta['exceedance_rate']:.0%} exceedance rate)"
        )

    parameters = parameter_groups(head, spec.weight_decay)
    optimiser = torch.optim.AdamW(parameters, lr=spec.learning_rate, betas=(0.9, 0.95))

    steps_per_epoch = len(train_loader)
    if spec.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, spec.max_steps_per_epoch)
    total_steps = steps_per_epoch * spec.epochs
    warmup_steps = int(steps_per_epoch * spec.warmup_epochs)

    run_dir = Path(spec.out_dir) / spec.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.csv"
    if not log_path.exists():
        with log_path.open("w", newline="") as handle:
            csv.writer(handle).writerow(
                ["epoch", "train_objective", "val_objective", "val_mae_mm", "seconds"]
            )

    best = float("inf")
    global_step = 0
    for epoch in range(1, spec.epochs + 1):
        train_set.set_epoch(epoch)
        started = time.perf_counter()
        running, seen = 0.0, 0

        for step, batch in enumerate(train_loader):
            if spec.max_steps_per_epoch and step >= spec.max_steps_per_epoch:
                break
            lr = learning_rate_at(
                global_step,
                total_steps,
                spec.learning_rate,
                warmup_steps,
                spec.min_lr_ratio,
            )
            for group in optimiser.param_groups:
                group["lr"] = lr

            x = flatten_time(batch["input"]).to(device, non_blocking=True)
            target_mm = to_mm(batch["target"][:, 0, channel].to(device))

            if spec.freeze_backbone:
                with torch.no_grad():  # no graph through a frozen encoder
                    tokens = backbone(x)
            else:
                tokens = backbone(x)

            loss, _ = score_batch(spec, head, tokens, target_mm, thresholds, weights)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            if spec.grad_clip:
                torch.nn.utils.clip_grad_norm_(head.parameters(), spec.grad_clip)
            optimiser.step()

            running += loss.detach().item()
            seen += 1
            global_step += 1
            if spec.log_every and step % spec.log_every == 0:
                print(
                    f"  epoch {epoch:>3} step {step:>5}/{steps_per_epoch} "
                    f"{spec.objective} {running / max(seen, 1):.4f} lr {lr:.2e}"
                )

        train_objective = running / max(seen, 1)
        val_objective, val_mae = validate(
            spec,
            backbone,
            head,
            val_loader,
            device,
            to_mm,
            channel,
            thresholds,
            weights,
        )
        seconds = time.perf_counter() - started
        print(
            f"epoch {epoch:>3}  train {train_objective:.4f}  "
            f"val {val_objective:.4f}  MAE {val_mae:.4f} mm  {seconds / 60:.1f} min"
        )
        with log_path.open("a", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    epoch,
                    f"{train_objective:.6f}",
                    f"{val_objective:.6f}",
                    f"{val_mae:.6f}",
                    f"{seconds:.1f}",
                ]
            )

        if val_objective < best:
            best = val_objective
            torch.save(
                {
                    "head": head.state_dict(),
                    "spec": asdict(spec),
                    "epoch": epoch,
                    "val_objective": val_objective,
                    "val_mae_mm": val_mae,
                    "variable": variable,
                    "data_config": str(data_config),
                },
                run_dir / "best.pt",
            )
            print(f"  new best: {best:.4f}")

    print(f"done. {run_dir}")
    return run_dir
