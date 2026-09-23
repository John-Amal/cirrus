"""Masked-autoencoder pretraining.

Written out rather than delegated to a framework, for the same reason the
attention was: everything that goes wrong in training goes wrong here.

Decisions worth knowing, because each one is a common way to get worse
results without any error appearing:

**Weight decay is applied selectively.** Biases, LayerNorm parameters,
position embeddings and the mask token are excluded. Decaying a LayerNorm
gain pulls it toward zero and fights the normalisation; decaying position
embeddings erodes exactly the information they exist to carry.

**Warmup then cosine decay.** Transformers are unstable in the first few
hundred steps, when attention is near-uniform and gradients are large.
Ramping the learning rate up over the first epochs, then decaying it
smoothly, is the standard recipe and it matters more than the peak value.

**Validation uses a fixed mask.** The masking is random, so a fresh draw each
epoch would make validation loss jump around for reasons unrelated to
learning. Seeding it means the validation task is literally the same problem
every time, and the curve reflects the model rather than the dice.

**Mixed precision only on CUDA.** Autocast on Apple's mps backend is not
consistently faster and has rough edges; float32 is the honest default on a
Mac. The flag exists so an HPC run can turn it on.
"""

from __future__ import annotations

import csv
import math
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
from cirrus.device import device_report, get_device
from cirrus.models.mae import MaeSpec, MaskedAutoencoder
from cirrus.models.patch_embed import flatten_time
from cirrus.models.vit import BackboneSpec, ViT


@dataclass(frozen=True)
class PretrainSpec:
    """Everything about a pretraining run that is not architecture."""

    name: str = "mae_small"
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1.5e-4
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    min_lr_ratio: float = 0.01
    grad_clip: float = 1.0
    num_workers: int = 0
    seed: int = 0
    amp: bool = False
    log_every: int = 50
    max_steps_per_epoch: int = 0  # 0 means the whole split
    val_batches: int = 20
    out_dir: str = "runs"

    @classmethod
    def from_yaml(cls, path: str | Path) -> PretrainSpec:
        """Load a spec, rejecting unknown keys."""
        raw: dict[str, Any] = read_yaml_mapping(path, (f.name for f in fields(cls)))
        return cls(**raw)


def learning_rate_at(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int,
    min_lr_ratio: float = 0.01,
) -> float:
    """Linear warmup, then cosine decay to ``min_lr_ratio * base_lr``."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    remaining = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / remaining)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Split parameters into decayed and undecayed groups.

    Matrices are decayed. Everything one-dimensional (biases, norm gains) is
    not, and neither are the position embeddings or mask token, which encode
    information rather than represent a linear map.
    """
    decayed: list[torch.nn.Parameter] = []
    undecayed: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        exempt = param.ndim < 2 or "pos_embed" in name or "mask_token" in name
        (undecayed if exempt else decayed).append(param)
    return [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": undecayed, "weight_decay": 0.0},
    ]


def dynamic_target_indices(
    n_steps: int, n_channels_per_step: int, n_dynamic: int
) -> list[int]:
    """Channel positions of the dynamic variables after time flattening.

    ``flatten_time`` is time-major, so step ``t`` occupies channels
    ``t * n_channels_per_step ...``, and within a step the dynamic channels
    come first, before statics and forcings.
    """
    return [
        step * n_channels_per_step + channel
        for step in range(n_steps)
        for channel in range(n_dynamic)
    ]


def build_model(
    dataset: ERA5Dataset,
    backbone_spec: BackboneSpec,
    mae_spec: MaeSpec,
    data_spec: IngestSpec,
) -> MaskedAutoencoder:
    """Assemble the MAE for a given dataset's channel layout."""
    sample = dataset[0]["input"]
    n_steps, n_per_step, height, width = sample.shape
    n_dynamic = len(data_spec.time_channels)

    backbone = ViT(
        in_channels=n_steps * n_per_step,
        spec=backbone_spec,
        grid=(height, width),
    )
    latitudes = None
    if mae_spec.latitude_weighted:
        store = xr.open_zarr(data_spec.output, chunks=None)
        # copy(): the array backing a zarr coordinate is read-only, and
        # torch refuses to share memory with a non-writable buffer.
        latitudes = torch.as_tensor(store["latitude"].values.copy())
    return MaskedAutoencoder(
        backbone=backbone,
        target_indices=dynamic_target_indices(n_steps, n_per_step, n_dynamic),
        spec=mae_spec,
        latitudes=latitudes,
    )


@torch.no_grad()
def validate(
    model: MaskedAutoencoder,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    max_batches: int,
    seed: int = 0,
) -> float:
    """Mean validation loss, with the masking held fixed across epochs."""
    model.eval()
    generator = torch.Generator(device="cpu")
    total, seen = 0.0, 0
    for index, batch in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        generator.manual_seed(seed + index)  # same mask for this batch, always
        x = flatten_time(batch["input"]).to(device)
        total += float(model(x, generator)["loss"])
        seen += 1
    model.train()
    return total / max(seen, 1)


def pretrain(
    spec: PretrainSpec,
    backbone_spec: BackboneSpec,
    mae_spec: MaeSpec,
    data_config: str | Path = "configs/data/era5_5625.yaml",
    resume: Path | None = None,
) -> Path:
    """Run pretraining and return the run directory."""
    torch.manual_seed(spec.seed)
    device = get_device()
    print(f"device: {device_report(device)}")

    data_spec = IngestSpec.from_yaml(data_config)
    train_set = ERA5Dataset.from_configs(split="train", data=data_config)
    val_set = ERA5Dataset.from_configs(split="val", data=data_config)
    print(f"samples: {len(train_set):,} train, {len(val_set):,} val")

    # persistent_workers keeps the worker processes alive between epochs.
    # Without it they are torn down and respawned every epoch, and macOS
    # spawns rather than forks: each worker re-imports torch and reopens the
    # zarr store, seconds of dead time per epoch. prefetch_factor lets each
    # worker stay several batches ahead so the GPU is not left waiting.
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

    model = build_model(train_set, backbone_spec, mae_spec, data_spec).to(device)
    print(f"model: {model.backbone.n_parameters:,} backbone parameters")
    print(f"masking: {model.n_visible}/{model.n_tokens} tokens visible")

    optimiser = torch.optim.AdamW(
        parameter_groups(model, spec.weight_decay),
        lr=spec.learning_rate,
        betas=(0.9, 0.95),
    )
    use_amp = spec.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    steps_per_epoch = len(train_loader)
    if spec.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, spec.max_steps_per_epoch)
    total_steps = steps_per_epoch * spec.epochs
    warmup_steps = int(steps_per_epoch * spec.warmup_epochs)

    run_dir = Path(spec.out_dir) / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.csv"
    start_epoch, best_val, global_step = 1, float("inf"), 0

    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        start_epoch = int(state["epoch"]) + 1
        best_val = float(state.get("best_val", float("inf")))
        global_step = int(state.get("global_step", 0))
        print(f"resumed from {resume} at epoch {start_epoch}")
    elif not log_path.exists():
        with log_path.open("w", newline="") as handle:
            csv.writer(handle).writerow(
                ["epoch", "train_loss", "val_loss", "lr", "seconds"]
            )

    for epoch in range(start_epoch, spec.epochs + 1):
        train_set.set_epoch(epoch)  # changes the augmentation draw
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
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = model(x)["loss"]

            scaler.scale(loss).backward()
            if spec.grad_clip:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), spec.grad_clip)
            scaler.step(optimiser)
            scaler.update()

            running += loss.detach().item()  # detach: keep the graph out of the log
            seen += 1
            global_step += 1
            if spec.log_every and step % spec.log_every == 0:
                print(
                    f"  epoch {epoch:>3} step {step:>5}/{steps_per_epoch} "
                    f"loss {running / max(seen, 1):.4f} lr {lr:.2e}"
                )

        train_loss = running / max(seen, 1)
        val_loss = validate(model, val_loader, device, spec.val_batches, spec.seed)
        seconds = time.perf_counter() - started
        print(
            f"epoch {epoch:>3}  train {train_loss:.4f}  val {val_loss:.4f}  "
            f"{seconds / 60:.1f} min"
        )
        with log_path.open("a", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{lr:.3e}",
                    f"{seconds:.1f}",
                ]
            )

        state = {
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val": min(best_val, val_loss),
            # Everything needed to rebuild the model in Phase 3.
            "pretrain_spec": asdict(spec),
            "backbone_spec": asdict(backbone_spec),
            "mae_spec": asdict(mae_spec),
            "data_config": str(data_config),
        }
        torch.save(state, run_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, run_dir / "best.pt")
            print(f"  new best: {best_val:.4f}")

    print(f"done. checkpoints and log in {run_dir}")
    return run_dir
