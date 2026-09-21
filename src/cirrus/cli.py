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

    from cirrus.train.loop import train  # imported late: torch is slow to load

    cfg = Config.from_yaml(args.config)
    if args.device is not None:
        cfg = Config(**{**cfg.to_dict(), "device": args.device})

    train(cfg, checkpoint_dir=args.checkpoint_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
