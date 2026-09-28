"""The figure that makes the Phase 3 result visible.

A table of amplitude ratios states that deterministic objectives lose
intensity. A picture shows it: the same event, predicted by heads that differ
only in their training objective.

The layout is deliberate. The top row is the truth and the two deterministic
arms. The bottom row is the same distributional model shown twice -- as its
predictive **mean**, and as a single **draw** -- beside the tail-weighted
arm's draw. The mean panel is the point: a distributional model asked for one
number is as smooth as a deterministic one. The sharpness is in the
distribution, not in the architecture, and it only appears when you sample
from it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from cirrus.data.dataset import ERA5Dataset
from cirrus.data.ingest import IngestSpec
from cirrus.data.normalise import Normaliser, NormaliseSpec
from cirrus.device import device_report, get_device
from cirrus.eval.compare import load_arm
from cirrus.models.patch_embed import flatten_time
from cirrus.models.vit import ViT
from cirrus.train.finetune import FinetuneSpec, PrecipitationTarget, load_backbone

PANELS: tuple[tuple[str, str, str], ...] = (
    ("truth", "", "ERA5 truth"),
    ("mse", "point", "MSE (point)"),
    ("l1", "point", "L1 (point)"),
    ("crps", "mean", "CRPS, predictive mean"),
    ("crps", "draw", "CRPS, single draw"),
    ("twcrps_p90", "draw", "tail-weighted CRPS, draw"),
)


def find_extreme_sample(
    loader: DataLoader[dict[str, torch.Tensor]],
    to_mm: PrecipitationTarget,
    channel: int,
    batches: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Pick the validation sample containing the heaviest precipitation.

    A quiet day would show nothing: every arm predicts light rain and the
    panels look alike. The interesting comparison needs an event.
    """
    best_input, best_target, best_value = None, None, -1.0
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        target = to_mm(batch["target"][:, 0, channel])
        peaks = target.flatten(1).max(dim=1).values
        position = int(peaks.argmax())
        if float(peaks[position]) > best_value:
            best_value = float(peaks[position])
            best_input = batch["input"][position : position + 1]
            best_target = target[position : position + 1]
    if best_input is None or best_target is None:
        raise ValueError("no samples found")
    return best_input, best_target, best_value


@torch.no_grad()
def comparison_figure(
    run_root: str | Path = "runs",
    out_path: str | Path = "docs/objective_comparison.png",
    split: str = "val",
    search_batches: int = 20,
    seed: int = 0,
    data_config: str | Path = "configs/data/era5_5625.yaml",
    normalise_config: str | Path = "configs/data/normalise.yaml",
    variable: str = "total_precipitation_6hr",
) -> Path:
    """Render truth beside each arm's prediction for one heavy event."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    device = get_device()
    print(f"device: {device_report(device)}")

    data_spec = IngestSpec.from_yaml(data_config)
    channel = data_spec.time_channels.index(variable)
    normaliser = Normaliser.load(NormaliseSpec.from_yaml(normalise_config).output)
    to_mm = PrecipitationTarget(normaliser, variable)

    dataset = ERA5Dataset.from_configs(split=split, data=data_config)
    loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        dataset, batch_size=32, shuffle=False
    )
    inputs, target, peak = find_extreme_sample(loader, to_mm, channel, search_batches)
    print(f"selected a sample peaking at {peak:.1f} mm / 6h")

    fields: dict[str, np.ndarray] = {
        "truth": target[0].cpu().numpy(),
    }
    backbone: ViT | None = None
    x = flatten_time(inputs).to(device)

    for arm, kind, _ in PANELS:
        if arm == "truth" or f"{arm}:{kind}" in fields:
            continue
        run_dir = Path(run_root) / f"finetune_{arm}"
        if not (run_dir / "best.pt").exists():
            print(f"skipping {arm}: no checkpoint")
            continue
        if backbone is None:
            state = torch.load(
                run_dir / "best.pt", map_location="cpu", weights_only=False
            )
            pretrained = FinetuneSpec(**state["spec"]).pretrained
            backbone, _ = load_backbone(pretrained, freeze=True)
            backbone = backbone.to(device).eval()

        spec, head = load_arm(run_dir, backbone, device)
        tokens = backbone(x)
        torch.manual_seed(seed)  # the same draw for every distributional arm
        if spec.is_distributional:
            samples = head(tokens).sample(spec.n_samples)
            fields[f"{arm}:mean"] = samples.mean(dim=-1)[0].cpu().numpy()
            fields[f"{arm}:draw"] = samples[..., 0][0].cpu().numpy()
        else:
            fields[f"{arm}:point"] = head(tokens)[0].cpu().numpy()

    top = float(np.quantile(fields["truth"], 0.999))
    # Precipitation spans orders of magnitude and is mostly near zero, so a
    # linear scale renders every panel uniformly pale. A square-root scale is
    # the convention for rainfall maps and keeps light and heavy both visible.
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=max(top, 1e-3))
    figure, axes = plt.subplots(2, 3, figsize=(15, 6.4))
    image = None

    for position, (arm, kind, title) in enumerate(PANELS):
        axis = axes.flat[position]
        key = "truth" if arm == "truth" else f"{arm}:{kind}"
        if key not in fields:
            axis.set_visible(False)
            continue
        field = fields[key]
        image = axis.imshow(field, origin="lower", cmap="Blues", norm=norm)
        ratio = float(np.quantile(field, 0.999) / max(top, 1e-9))
        label = title if arm == "truth" else f"{title}\np99.9 of panel: {ratio:.2f}x"
        axis.set_title(label, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])

    if image is not None:
        figure.colorbar(
            image, ax=axes, fraction=0.02, pad=0.01, label="precipitation [mm / 6h]"
        )
    figure.suptitle(
        "Same event, same backbone, same data — only the training objective differs",
        fontsize=12,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    print(f"figure: {out_path}")
    return out_path


def figure_metadata(out_path: str | Path) -> dict[str, Any]:
    """Where the figure came from, for the caption."""
    return {"path": str(out_path), "panels": [title for _, _, title in PANELS]}
