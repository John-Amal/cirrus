# cirrus

A small weather foundation model, built end to end: self-supervised
pretraining on ERA5, fine-tuned downstream heads, benchmarking against NWP
baselines, and containerised inference.

[![ci](https://github.com/John-Amal/cirrus/actions/workflows/ci.yml/badge.svg)](https://github.com/John-Amal/cirrus/actions/workflows/ci.yml)

> **Status: Phase 0 — scaffolding.** The plumbing works end to end on a dummy
> model. No science in here yet.

## Why

Large weather models (GraphCast, Aurora, AIFS) are trained by teams with
substantial compute. This repository implements the same pipeline at a scale
one person can train and, more importantly, fully explain: every component
from the data loader to the attention mechanism is written out rather than
imported.

The distinguishing piece is the treatment of extremes. Standard training
objectives optimise mean skill and are indifferent to the tail of the
distribution, which is precisely the part that matters for risk. One of the
fine-tuning heads uses an objective informed by extreme value theory, and
evaluation includes return-level accuracy alongside conventional scores.

## Quickstart

```bash
git clone https://github.com/John-Amal/cirrus.git
cd cirrus
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cirrus device              # report the detected device
cirrus train --config configs/tiny.yaml
```

The `tiny` config trains in seconds on any machine, including CPU-only CI. It
exists so that every change can be validated before anything long-running
starts.

## Roadmap

| Phase | Content | Status |
| --- | --- | --- |
| 0 | Scaffolding, CI, training loop on synthetic data | done |
| 1 | ERA5 data pipeline and augmentation | next |
| 2 | Masked-autoencoder pretraining (ViT backbone) | |
| 3 | Fine-tuning: forecasting, downscaling, extremes | |
| 4 | Benchmarking, ablations, scaling behaviour | |
| 5 | Export, serving, LLM agent interface | |

## Development

```bash
pre-commit install   # format and lint on every commit
pytest               # run the test suite
mypy src             # type check
```

Design decisions and negative results are recorded in
[`docs/experiments.md`](docs/experiments.md).

## Related work by the author

- [Climate Risk Explorer](https://github.com/John-Amal) — end-to-end Swiss
  climate risk pipeline and dashboard.
- `catagg` — open-source catastrophe loss aggregation (ELT to OEP/AEP curves).

## Licence

MIT.
