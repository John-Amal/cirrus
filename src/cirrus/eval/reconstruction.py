"""Diagnostics for a pretrained model.

Two things a single loss number cannot tell you.

**Which variables it learned.** The headline loss averages over every channel,
and they will not be equal. Geopotential is smooth and large-scale and should
reconstruct well; precipitation is intermittent, skewed and small-scale, and
will almost certainly be the worst by a wide margin. That gap is the tail
problem showing up in pretraining, before any extremes head exists.

**Whether the reconstructions look like weather.** A model can reach a
respectable MSE by predicting something close to climatology everywhere:
blurry, structureless, and wrong in exactly the way that matters for
extremes. Looking at the fields catches that immediately; the loss does not.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from cirrus.data.dataset import ERA5Dataset
from cirrus.data.ingest import IngestSpec
from cirrus.data.normalise import Normaliser
from cirrus.models.mae import MaeSpec, MaskedAutoencoder
from cirrus.models.patch_embed import flatten_time, unpatchify
from cirrus.models.vit import BackboneSpec
from cirrus.train.pretrain import build_model

# Display conversions, applied after denormalisation back to physical units.
UNITS: dict[str, tuple[float, str]] = {
    "total_precipitation_6hr": (1000.0, "mm / 6h"),
    "mean_sea_level_pressure": (0.01, "hPa"),
}


def to_physical(
    field: torch.Tensor, normaliser: Normaliser, scale: float
) -> np.ndarray:
    """Denormalise one channel and apply its display scaling."""
    values = normaliser.denormalise(field[None, None].numpy())[0, 0]
    return cast(np.ndarray, values * scale)


def load_pretrained(
    checkpoint: str | Path, split: str = "val"
) -> tuple[MaskedAutoencoder, ERA5Dataset, IngestSpec]:
    """Rebuild the model recorded in a checkpoint and load its weights.

    The checkpoint carries its own architecture and objective specs, so a
    model is never reconstructed from a config file that may have changed
    since the run.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    data_config = state.get("data_config", "configs/data/era5_5625.yaml")
    data_spec = IngestSpec.from_yaml(data_config)
    dataset = ERA5Dataset.from_configs(split=split, data=data_config)

    model = build_model(
        dataset,
        BackboneSpec(**state["backbone_spec"]),
        MaeSpec(**state["mae_spec"]),
        data_spec,
    )
    model.load_state_dict(state["model"])
    model.eval()
    return model, dataset, data_spec


@torch.no_grad()
def per_channel_loss(
    model: MaskedAutoencoder,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    max_batches: int = 20,
    seed: int = 0,
) -> dict[str, float]:
    """Masked reconstruction MSE per variable, averaged over input steps.

    Uses the same fixed masking as validation, so the numbers are comparable
    between checkpoints.
    """
    weights = cast(torch.Tensor, model.token_weights).to(device)
    n_channels = model.n_target_channels
    values_per_channel = model.patch * model.patch

    totals = torch.zeros(n_channels, device=device)
    denominator = torch.zeros((), device=device)
    generator = torch.Generator(device="cpu")

    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        generator.manual_seed(seed + index)
        x = flatten_time(batch["input"]).to(device)
        out = model(x, generator)

        shape = (x.shape[0], model.n_tokens, n_channels, values_per_channel)
        error = (
            (out["prediction"].reshape(shape) - model.targets(x).reshape(shape)) ** 2
        ).mean(-1)
        weighted = (out["mask"] * weights).unsqueeze(-1)  # (B, N, 1)
        totals += (error * weighted).sum(dim=(0, 1))
        denominator += weighted.sum()

    per_channel = (totals / denominator.clamp(min=1e-8)).cpu().numpy()
    return dict(zip(model_channel_names(model), per_channel.tolist(), strict=True))


def model_channel_names(model: MaskedAutoencoder) -> list[str]:
    """Names of the reconstructed channels, one per target index."""
    indices = cast(torch.Tensor, model.target_indices).tolist()
    return [f"channel_{i}" for i in indices]


def variable_losses(
    channel_losses: dict[str, float], variables: tuple[str, ...]
) -> dict[str, float]:
    """Collapse per-channel losses onto variable names.

    Target channels run variable-fastest within each input step, so channel
    ``k`` belongs to variable ``k % len(variables)``.
    """
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for position, value in enumerate(channel_losses.values()):
        grouped[variables[position % len(variables)]].append(value)
    return {name: float(np.mean(values)) for name, values in grouped.items()}


def distribution_summary(
    predicted: np.ndarray, truth: np.ndarray, quantiles: tuple[float, ...]
) -> dict[str, dict[str, float]]:
    """Mean, quantiles and maximum of two samples, with their ratios."""
    report: dict[str, dict[str, float]] = {
        "mean": {"truth": float(truth.mean()), "prediction": float(predicted.mean())}
    }
    for q in quantiles:
        report[f"p{q * 100:g}"] = {
            "truth": float(np.quantile(truth, q)),
            "prediction": float(np.quantile(predicted, q)),
        }
    report["max"] = {"truth": float(truth.max()), "prediction": float(predicted.max())}
    for row in report.values():
        row["ratio"] = (
            row["prediction"] / row["truth"] if row["truth"] else float("nan")
        )
    return report


@torch.no_grad()
def amplitude_deficit(
    model: MaskedAutoencoder,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    normaliser: Normaliser,
    variable_index: int,
    n_variables: int,
    max_batches: int = 20,
    seed: int = 0,
    quantiles: tuple[float, ...] = (0.9, 0.99, 0.999),
) -> dict[str, dict[str, dict[str, float]]]:
    """Compare reconstructed and true value distributions over masked patches.

    Reported in two spaces, because they answer different questions.

    **Transformed** is the space the loss is computed in: z-scored, and for
    precipitation, after ``log1p``. A mean ratio near 1 here says the model
    is unbiased at the thing it was actually optimised for.

    **Physical** is millimetres. If the mean ratio is below 1 here while it
    is near 1 above, the shortfall comes from the transform rather than the
    model: ``expm1`` is convex, so by Jensen's inequality transforming a
    conditional mean back gives less than the mean of the transformed values.
    An MSE objective in log space is therefore systematically dry in mm.

    The tail ratios measure the other effect: squared error is minimised by
    predicting the conditional mean, so a model that cannot place an event
    exactly is rewarded for spreading it out, and intensity collapses.
    """
    generator = torch.Generator(device="cpu")
    values_per_patch = model.patch * model.patch
    predicted_parts, true_parts = [], []

    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        generator.manual_seed(seed + index)
        x = flatten_time(batch["input"]).to(device)
        out = model(x, generator)

        shape = (x.shape[0], model.n_tokens, model.n_target_channels, values_per_patch)
        prediction = out["prediction"].reshape(shape)
        target = model.targets(x).reshape(shape)
        hidden = out["mask"].bool()

        # The variable appears once per input step; pool them.
        for channel in range(variable_index, model.n_target_channels, n_variables):
            predicted_parts.append(prediction[:, :, channel][hidden].cpu().numpy())
            true_parts.append(target[:, :, channel][hidden].cpu().numpy())

    predicted_z = np.concatenate(predicted_parts)
    truth_z = np.concatenate(true_parts)

    # Undo only the z-scoring: this is log1p space for a log channel.
    mean, std = float(normaliser.mean[0]), float(normaliser.std[0])
    transformed = distribution_summary(
        predicted_z * std + mean, truth_z * std + mean, quantiles
    )

    def to_units(values: np.ndarray) -> np.ndarray:
        return cast(
            np.ndarray, normaliser.denormalise(values.reshape(-1, 1, 1, 1)).reshape(-1)
        )

    physical = distribution_summary(to_units(predicted_z), to_units(truth_z), quantiles)
    return {"transformed": transformed, "physical": physical}


@torch.no_grad()
def reconstruction_figure(
    model: MaskedAutoencoder,
    dataset: ERA5Dataset,
    variables: tuple[str, ...],
    channels: list[str],
    out_path: str | Path,
    sample_index: int = 0,
    seed: int = 0,
) -> Path:
    """Write a truth / masked input / reconstruction figure.

    Masked patches are drawn as blank rather than zero: zero is the mean
    after normalisation, and painting the mean into the gaps makes a model
    look better than it is.
    """
    import matplotlib

    matplotlib.use("Agg")  # no display in a terminal session
    import matplotlib.pyplot as plt

    item = dataset[sample_index]
    x = flatten_time(item["input"].unsqueeze(0))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    out = model(x, generator)
    target, prediction, mask = model.targets(x), out["prediction"], out["mask"]
    filled = target * (1 - mask.unsqueeze(-1)) + prediction * mask.unsqueeze(-1)

    grid, n_channels = model.grid, model.n_target_channels
    truth = unpatchify(target, model.patch, grid, n_channels)[0]
    reconstruction = unpatchify(filled, model.patch, grid, n_channels)[0]
    width = target.shape[-1]
    mask_field = unpatchify(
        mask.unsqueeze(-1).expand(-1, -1, width), model.patch, grid, n_channels
    )[0]

    normaliser = dataset.source.dynamic_norm
    figure, axes = plt.subplots(
        len(channels), 3, figsize=(13, 3.1 * len(channels)), squeeze=False
    )

    for row, name in enumerate(channels):
        position = variables.index(name)  # first input step
        scale, unit = UNITS.get(name, (1.0, ""))
        single = normaliser.subset([name])

        true_field = to_physical(truth[position], single, scale)
        recon_field = to_physical(reconstruction[position], single, scale)
        hidden = to_physical(truth[position], single, scale)
        hidden[mask_field[position].numpy() > 0.5] = np.nan

        low, high = np.percentile(true_field, [1, 99])
        panels = [
            (true_field, "truth"),
            (hidden, "masked input"),
            (recon_field, "reconstruction"),
        ]
        for column, (field, title) in enumerate(panels):
            axis = axes[row][column]
            image = axis.imshow(
                field, origin="lower", cmap="RdBu_r", vmin=low, vmax=high
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(title)
            if column == 0:
                label = f"{name}\n[{unit}]" if unit else name
                axis.set_ylabel(label, fontsize=8)
        figure.colorbar(image, ax=axes[row], fraction=0.015, pad=0.01)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return out_path


def inspect(
    checkpoint: str | Path,
    channels: list[str],
    out_path: str | Path,
    batches: int = 20,
    split: str = "val",
    sample_index: int = 0,
    amplitude_variable: str = "total_precipitation_6hr",
) -> dict[str, float]:
    """Score a checkpoint per variable and write a reconstruction figure."""
    from cirrus.device import device_report, get_device

    model, dataset, data_spec = load_pretrained(checkpoint, split)
    device = get_device()
    print(f"device: {device_report(device)}")
    model = model.to(device)

    loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        dataset, batch_size=16, shuffle=False
    )
    channel_losses = per_channel_loss(model, loader, device, batches)
    variables = data_spec.time_channels
    by_variable = variable_losses(channel_losses, variables)

    print(f"\nmasked reconstruction MSE by variable ({batches} {split} batches)")
    print(f"{'variable':32s} {'MSE':>8s}")
    for name, value in sorted(by_variable.items(), key=lambda kv: -kv[1]):
        print(f"{name:32s} {value:8.4f}")

    if amplitude_variable in variables:
        scale, unit = UNITS.get(amplitude_variable, (1.0, "native units"))
        report = amplitude_deficit(
            model=model,
            loader=loader,
            device=device,
            normaliser=dataset.source.dynamic_norm.subset([amplitude_variable]),
            variable_index=variables.index(amplitude_variable),
            n_variables=len(variables),
            max_batches=batches,
        )
        for space, rows in report.items():
            factor = scale if space == "physical" else 1.0
            label = unit if space == "physical" else "log1p space, where the loss lives"
            print(f"\n{amplitude_variable} over masked patches [{label}]")
            print(f"{'':10s} {'truth':>10s} {'predicted':>10s} {'ratio':>8s}")
            for name, row in rows.items():
                print(
                    f"{name:10s} {row['truth'] * factor:10.3f} "
                    f"{row['prediction'] * factor:10.3f} {row['ratio']:8.2f}"
                )

    figure_path = reconstruction_figure(
        model.cpu(), dataset, variables, channels, out_path, sample_index
    )
    print(f"\nfigure: {figure_path}")
    return by_variable


def specs_from_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
    """Architecture, objective and run settings recorded in a checkpoint."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return {
        "backbone": state["backbone_spec"],
        "objective": state["mae_spec"],
        "run": state["pretrain_spec"],
        "epoch": state["epoch"],
        "best_val": state.get("best_val"),
    }
