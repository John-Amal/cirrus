"""Prepare and publish the pretrained encoder to the Hugging Face Hub.

Builds a release directory, then uploads it only if asked. Two deliberate
choices:

- **The optimiser state is stripped.** A training checkpoint carries AdamW's
  moments, which are roughly twice the size of the weights and useless to
  anyone downloading the model.
- **The encoder is saved separately** from the full autoencoder, so somebody
  who only wants the backbone does not have to know how the MAE was wired.
  The decoder exists to make pretraining work and has no use afterwards.

Usage::

    python scripts/publish_to_hub.py --checkpoint runs/mae_small/best.pt
    python scripts/publish_to_hub.py --checkpoint runs/mae_small/best.pt --push

Requires ``pip install huggingface_hub`` and ``huggingface-cli login``
(``hf auth login`` in newer versions) before pushing.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import torch

from cirrus.data.ingest import IngestSpec
from cirrus.data.windows import FORCING_CHANNELS, WindowSpec

DEFAULT_REPO = "John-Amal/cirrus-mae-5625"


def build_release(
    checkpoint: Path, out_dir: Path, card: Path, figure: Path | None
) -> Path:
    """Write the release directory and return its path."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    data_spec = IngestSpec.from_yaml(
        state.get("data_config", "configs/data/era5_5625.yaml")
    )
    window = WindowSpec.from_yaml("configs/data/windows.yaml")

    per_step = (
        *data_spec.time_channels,
        *data_spec.static_variables,
        *FORCING_CHANNELS,
    )
    input_channels = [
        f"{name}_t{step}" for step in range(window.n_input) for name in per_step
    ]

    backbone = {
        key[len("backbone.") :]: value
        for key, value in state["model"].items()
        if key.startswith("backbone.")
    }
    if not backbone:
        raise ValueError("no backbone weights found in the checkpoint")

    release: dict[str, Any] = {
        "backbone": backbone,
        "model": state["model"],
        "backbone_spec": state["backbone_spec"],
        "mae_spec": state["mae_spec"],
        "input_channels": input_channels,
        "dynamic_channels": list(data_spec.time_channels),
        "grid": [32, 64],
        "epoch": state["epoch"],
        "best_val": state.get("best_val"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "cirrus-mae-5625.pt"
    torch.save(release, weights_path)
    shutil.copy(card, out_dir / "README.md")  # the Hub renders README.md as the card
    if figure is not None and figure.exists():
        shutil.copy(figure, out_dir / "reconstruction.png")

    training_mb = checkpoint.stat().st_size / 1e6
    release_mb = weights_path.stat().st_size / 1e6
    print(f"training checkpoint: {training_mb:6.1f} MB")
    print(f"release weights:     {release_mb:6.1f} MB  ({len(backbone)} tensors)")
    print(f"input channels:      {len(input_channels)}")
    print(f"release directory:   {out_dir}")
    return out_dir


def push(out_dir: Path, repo_id: str, private: bool) -> None:
    """Create the repository if needed and upload the release directory."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id, repo_type="model")
    print(f"pushed to https://huggingface.co/{repo_id}")


def main() -> int:
    """Prepare the release directory, and upload it when asked."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("runs/mae_small/best.pt")
    )
    parser.add_argument("--out", type=Path, default=Path("release"))
    parser.add_argument("--card", type=Path, default=Path("docs/MODEL_CARD.md"))
    parser.add_argument(
        "--figure", type=Path, default=Path("runs/mae_small/reconstruction.png")
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--push", action="store_true", help="upload; without it, only prepare locally"
    )
    args = parser.parse_args()

    out_dir = build_release(args.checkpoint, args.out, args.card, args.figure)
    if args.push:
        push(out_dir, args.repo, args.private)
    else:
        print("\nprepared only. Review the files, then re-run with --push")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
