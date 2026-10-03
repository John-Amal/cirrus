"""Publish everything the inference service needs to the Hugging Face Hub.

The exported model is useless on its own. The service normalises incoming
fields with the statistics the model was trained on, and reports exceedance
probabilities against per-cell climatological thresholds. Ship the model
without those and the service either fails or, worse, answers confidently
with numbers derived from the wrong normalisation.

So the bundle is the unit: model, normalisation statistics, thresholds, and
a record of which run produced them.

Usage::

    python scripts/publish_serving_bundle.py --arm twcrps_p90
    python scripts/publish_serving_bundle.py --arm twcrps_p90 --push

Requires ``pip install -e ".[hub]"`` and ``hf auth login``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

DEFAULT_REPO = "John-Amal/cirrus-serving-5625"

CARD = """---
license: mit
library_name: pytorch
tags:
  - weather
  - precipitation
  - probabilistic-forecasting
  - extreme-value-theory
---

# cirrus serving bundle

Everything the [`cirrus`](https://github.com/John-Amal/cirrus) inference
service needs: a TorchScript precipitation model and the statistics it was
trained with.

The model maps two timesteps of normalised ERA5 fields to the parameters of a
**censored shifted gamma** per grid cell — `Y = max(0, Gamma(shape, scale) +
shift)` — from which expected rainfall, exceedance probabilities and
arbitrary quantiles follow in closed form.

| file | what it is |
|---|---|
| `{arm}.pt` | TorchScript predictor, verified bit-identical to the PyTorch original |
| `normalisation.json` | Per-channel statistics from the 1979–2014 training years |
| `thresholds.json` | Per-cell exceedance thresholds, 2% of 6-hourly steps |
| `bundle.json` | Which run produced this, and the channel order the model expects |

## Why the statistics ship with the model

Inputs must be normalised with the same statistics used in training.
Different ones produce plausible-looking nonsense rather than an error, so
they travel together rather than being left to the caller.

## Use

The service in the repository loads this bundle directly:

```bash
pip install "cirrus[serve] @ git+https://github.com/John-Amal/cirrus"
cirrus serve --arm {arm}
```

Trained on ERA5 via WeatherBench 2. Contains modified Copernicus Climate
Change Service information (1979–2022).
"""


def build_bundle(arm: str, out_dir: Path, repo: str) -> dict[str, Any]:
    """Assemble the serving bundle locally and return its manifest."""
    sources = {
        f"{arm}.pt": Path("serve") / f"{arm}.pt",
        "normalisation.json": Path("data/stats/era5_5625_train.json"),
        "thresholds.json": Path("data/stats/thresholds_train.json"),
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise SystemExit(
            "missing: "
            + ", ".join(missing)
            + "\nrun 'cirrus export', 'cirrus stats' and 'cirrus thresholds' first"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, path in sources.items():
        shutil.copy(path, out_dir / name)

    normalisation = json.loads((out_dir / "normalisation.json").read_text())
    thresholds = json.loads((out_dir / "thresholds.json").read_text())
    manifest = {
        "arm": arm,
        "repo": repo,
        "grid": [32, 64],
        "input_steps": 2,
        "channel_order": normalisation["channels"],
        "distribution": "censored shifted gamma",
        "outputs": ["shape", "scale", "shift"],
        "training_period": normalisation.get("meta", {}).get("train"),
        "threshold_rate": thresholds.get("meta", {}).get("exceedance_rate"),
    }
    (out_dir / "bundle.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "README.md").write_text(CARD.format(arm=arm))

    total = sum(path.stat().st_size for path in out_dir.iterdir())
    n_files = len(list(out_dir.iterdir()))
    print(f"bundle: {out_dir} ({total / 1e6:.1f} MB, {n_files} files)")
    for path in sorted(out_dir.iterdir()):
        print(f"  {path.name:24s} {path.stat().st_size / 1e6:7.2f} MB")
    return manifest


def push(out_dir: Path, repo: str, private: bool) -> None:
    """Create the repository if needed and upload the bundle."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo, repo_type="model")
    print(f"pushed to https://huggingface.co/{repo}")


def main() -> int:
    """Build the bundle, and upload it when asked."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="twcrps_p90")
    parser.add_argument("--out", type=Path, default=Path("release/serving"))
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    build_bundle(args.arm, args.out, args.repo)
    if args.push:
        push(args.out, args.repo, args.private)
    else:
        print("\nprepared only. Review the files, then re-run with --push")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
