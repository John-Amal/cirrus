"""Command line interface.

Runs are launched as ``cirrus train --config configs/tiny.yaml`` rather than
by editing constants in a script. Anything you can express as a flag is
something you can put in an experiment log.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cirrus.config import Config


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the requested command."""
    parser = argparse.ArgumentParser(prog="cirrus")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train a model")
    train_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tiny.yaml"),
        help="path to the run config",
    )
    train_parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "mps", "cuda"],
        help="override device autodetection",
    )
    train_parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="where to write the final checkpoint",
    )

    subparsers.add_parser("device", help="report the detected device")

    ingest_parser = subparsers.add_parser(
        "ingest", help="download an ERA5 subset to a local zarr store"
    )
    ingest_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/era5_5625.yaml"),
        help="path to the ingest spec",
    )
    ingest_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the spec against the source and report size; download nothing",
    )

    stats_parser = subparsers.add_parser(
        "stats", help="compute normalisation statistics over the training period"
    )
    stats_parser.add_argument(
        "--data", type=Path, default=Path("configs/data/era5_5625.yaml")
    )
    stats_parser.add_argument(
        "--splits", type=Path, default=Path("configs/data/splits.yaml")
    )
    stats_parser.add_argument(
        "--normalise", type=Path, default=Path("configs/data/normalise.yaml")
    )

    sample_parser = subparsers.add_parser("sample", help="inspect one training sample")
    sample_parser.add_argument(
        "--split", default="train", choices=["train", "val", "test"]
    )
    sample_parser.add_argument("--index", type=int, default=0)
    sample_parser.add_argument(
        "--data", type=Path, default=Path("configs/data/era5_5625.yaml")
    )
    sample_parser.add_argument(
        "--splits", type=Path, default=Path("configs/data/splits.yaml")
    )
    sample_parser.add_argument(
        "--window", type=Path, default=Path("configs/data/windows.yaml")
    )
    sample_parser.add_argument(
        "--normalise", type=Path, default=Path("configs/data/normalise.yaml")
    )

    pre = subparsers.add_parser("pretrain", help="masked-autoencoder pretraining")
    pre.add_argument("--config", type=Path, default=Path("configs/train/pretrain.yaml"))
    pre.add_argument("--model", type=Path, default=Path("configs/model/mae_small.yaml"))
    pre.add_argument(
        "--objective", type=Path, default=Path("configs/train/mae_objective.yaml")
    )
    pre.add_argument("--data", type=Path, default=Path("configs/data/era5_5625.yaml"))
    pre.add_argument("--resume", type=Path, default=None)
    pre.add_argument("--epochs", type=int, default=None, help="override the config")
    pre.add_argument(
        "--max-steps", type=int, default=None, help="steps per epoch; 0 means all"
    )

    args = parser.parse_args(argv)

    if args.command == "device":
        from cirrus.device import device_report, get_device

        print(device_report(get_device()))
        return 0

    if args.command == "ingest":
        # Imported late: keeps `cirrus device` fast and torch-free paths light.
        from cirrus.data.ingest import (
            IngestSpec,
            check_available,
            describe,
            ingest,
            open_source,
        )

        try:
            spec = IngestSpec.from_yaml(args.config)
            source = open_source(spec.source)
            check_available(source, spec)
        except ValueError as err:
            print(f"error: {err}", file=sys.stderr)
            return 1
        print(describe(source, spec))
        if not args.dry_run:
            ingest(spec, source=source)
        return 0

    if args.command == "stats":
        from cirrus.data.ingest import IngestSpec
        from cirrus.data.normalise import NormaliseSpec, compute_stats
        from cirrus.data.splits import Splits

        try:
            spec = IngestSpec.from_yaml(args.data)
            splits = Splits.from_yaml(args.splits)
            nspec = NormaliseSpec.from_yaml(args.normalise)
            norm = compute_stats(spec.output, spec, splits.train, nspec)
        except ValueError as err:
            print(f"error: {err}", file=sys.stderr)
            return 1
        path = norm.save(nspec.output)
        print(f"\n{'channel':32s} {'mean':>14s} {'std':>14s}")
        for name, m, s in zip(norm.channels, norm.mean, norm.std, strict=True):
            flag = "  (log)" if name in norm.log_channels else ""
            print(f"{name:32s} {m:14.6g} {s:14.6g}{flag}")
        print(f"\nwritten: {path}")
        return 0

    if args.command == "sample":
        import numpy as np

        from cirrus.data.dataset import ERA5Dataset

        try:
            dataset = ERA5Dataset.from_configs(
                split=args.split,
                data=args.data,
                splits=args.splits,
                window=args.window,
                normalise=args.normalise,
            )
        except (ValueError, FileNotFoundError) as err:
            print(f"error: {err}", file=sys.stderr)
            return 1

        item = dataset[args.index]
        when = np.array(item["time"].numpy()).astype("datetime64[ns]")
        print(f"split:     {args.split} ({len(dataset):,} samples)")
        print(f"sample:    {args.index}, last input step {when}")
        print(f"input:     {tuple(item['input'].shape)}")
        if "target" in item:
            print(f"target:    {tuple(item['target'].shape)}")
        print(f"channels:  {', '.join(dataset.input_channels)}")
        x = item["input"]
        print(
            f"input mean {x.mean():.3f}  std {x.std():.3f}  "
            f"min {x.min():.2f}  max {x.max():.2f}"
        )
        return 0

    if args.command == "pretrain":
        from dataclasses import replace

        from cirrus.models.mae import MaeSpec
        from cirrus.models.vit import BackboneSpec
        from cirrus.train.pretrain import PretrainSpec, pretrain

        try:
            # Distinct name: `spec` is bound to an IngestSpec in the branches
            # above, and a function-scoped name carries one type throughout.
            train_spec = PretrainSpec.from_yaml(args.config)
            backbone_spec = BackboneSpec.from_yaml(args.model)
            mae_spec = MaeSpec.from_yaml(args.objective)
        except ValueError as err:
            print(f"error: {err}", file=sys.stderr)
            return 1

        # Named keywords rather than **kwargs: dataclasses.replace cannot be
        # type-checked through an unpacked dict, and a suppression comment
        # would then flip between needed and unused as other things change.
        if args.epochs is not None:
            train_spec = replace(train_spec, epochs=args.epochs)
        if args.max_steps is not None:
            train_spec = replace(train_spec, max_steps_per_epoch=args.max_steps)

        pretrain(train_spec, backbone_spec, mae_spec, args.data, args.resume)
        return 0

    from cirrus.train.loop import train  # imported late: torch is slow to load

    cfg = Config.from_yaml(args.config)
    if args.device is not None:
        cfg = Config(**{**cfg.to_dict(), "device": args.device})

    train(cfg, checkpoint_dir=args.checkpoint_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
