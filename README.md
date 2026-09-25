# cirrus

A small weather foundation model, built end to end — and used to study one
thing the large ones get wrong: **the tail**.

[![ci](https://github.com/John-Amal/cirrus/actions/workflows/ci.yml/badge.svg)](https://github.com/John-Amal/cirrus/actions/workflows/ci.yml)
[![weights](https://img.shields.io/badge/%F0%9F%A4%97%20weights-cirrus--mae--5625-blue)](https://huggingface.co/John-Amal/cirrus-mae-5625)

> **Status: Phase 2 complete.** A 5M-parameter ViT is pretrained on 36 years
> of ERA5 with a masked-autoencoder objective, in under three hours on a
> laptop. Weights are on the
> [Hugging Face Hub](https://huggingface.co/John-Amal/cirrus-mae-5625).

![reconstruction](docs/reconstruction.png)

## The result so far

The model reconstructs hidden patches of the atmosphere well: validation MSE
**0.2385** in normalised units, against ~1.0 for predicting the mean. Skill
tracks atmospheric predictability — geopotential at 250 hPa reaches 0.021,
while precipitation is worst at 0.618, a thirty-fold spread.

The interesting number is what happens to extreme precipitation:

| ratio, predicted / true | log1p space (where the loss is computed) | physical (mm / 6h) |
|---|---|---|
| mean | **1.00** | 0.76 |
| p99 | 0.61 | 0.38 |
| p99.9 | 0.60 | **0.30** |

Measured over masked patches. The model is *exactly unbiased* in the space it
was optimised in, and reproduces **30% of the intensity** of the most extreme
events in millimetres.

Two mechanisms compound. Squared error is minimised by predicting the
conditional mean, so a model that cannot place an event precisely is rewarded
for spreading it out. That under-dispersion is then amplified by the inverse
transform: at the 99.9th percentile a ~1.0 log-unit error — unremarkable to
the loss — is the difference between 12.1 mm and 3.7 mm of rain.

**The loss function is nearly flat exactly where the extremes are.** That is
a property of the objective rather than of this checkpoint, and it motivates
everything in Phase 3.

## Why this project

Large weather models (GraphCast, Aurora, AIFS) are trained by teams with
substantial compute. This repository implements the same pipeline at a scale
one person can train and, more importantly, fully explain: patch embedding,
attention, the masking scheme and the training loop are written out rather
than imported, and checked against reference implementations.

The distinguishing piece is the treatment of extremes. Published evaluations
find that AI weather models underestimate both the frequency and intensity of
record-breaking events, and ECMWF attributes AIFS's under-prediction of heavy
precipitation partly to smoothing from its MSE loss. Work adapting foundation
models for extremes mostly still trains with bulk losses. `cirrus` tests
whether an extreme-value-informed objective changes that, at a scale where
the experiment can actually be run.

## Quickstart

```bash
git clone https://github.com/John-Amal/cirrus.git && cd cirrus
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cirrus device                                    # cuda -> mps -> cpu
cirrus ingest --config configs/data/era5_5625.yaml --dry-run
```

The full pipeline, from nothing to a trained model:

```bash
cirrus ingest    # ~11 GB of ERA5 from WeatherBench 2, resumable
cirrus stats     # latitude-weighted normalisation, training years only
cirrus pretrain  # ~8 min/epoch on an M-series laptop
cirrus inspect   # per-variable scores and the reconstruction figure
```

## How it works

| Stage | What it does |
|---|---|
| `data/ingest` | Pulls ERA5 at 5.625°, validating variables before downloading; resumable |
| `data/normalise` | Latitude-weighted, training-years-only statistics; `log1p` for precipitation |
| `data/windows` | Samples that never cross split boundaries or time gaps |
| `data/augment` | Longitude roll, paired with a solar-time shift so the diurnal cycle stays honest |
| `models/attention` | Multi-head attention written out, tested against PyTorch's fused version |
| `models/mae` | 75% masking, lightweight decoder, latitude-weighted loss on hidden patches only |
| `eval/reconstruction` | Per-variable scores and the tail-amplitude diagnostic |

At 5.625° the grid is 32×64, so 4×4 patches give 128 tokens — small enough
for full attention and for a laptop.

## Roadmap

| Phase | Content | Status |
| --- | --- | --- |
| 0 | Scaffolding, CI, training loop on synthetic data | done |
| 1 | ERA5 pipeline, normalisation, windowing, augmentation | done |
| 2 | Masked-autoencoder pretraining, published weights | done |
| 3 | Fine-tuning: forecasting baseline, extremes head with a GPD-informed loss | next |
| 4 | Benchmarking, ablations, out-of-distribution evaluation | |
| 5 | Export, serving, LLM agent interface | |

Phase 3 compares MSE, L1, CRPS and a GPD-informed objective on the same
frozen backbone. The 0.30 ratio above is the number to beat.

## Development

```bash
pre-commit install
ruff check . && mypy src && pytest    # what CI runs, in order
```

The tests worth reading assert properties rather than absence of crashes:
that the hand-written attention matches PyTorch's fused implementation, that
patch ordering agrees between the tokeniser and the loss targets, that the
longitude roll preserves local solar time, and that a sample window never
crosses a train/test boundary.

Design decisions and negative results, including the bugs, are recorded in
[`docs/experiments.md`](docs/experiments.md).

## Data

ERA5 reanalysis via [WeatherBench 2](https://weatherbench2.readthedocs.io/),
conservatively regridded to 64×32. Contains modified Copernicus Climate
Change Service information (1979–2014); neither the European Commission nor
ECMWF is responsible for any use of it.

## Related work by the author

- [Climate Risk Explorer](https://github.com/John-Amal) — end-to-end Swiss
  climate risk pipeline and dashboard.
- `catagg` — open-source catastrophe loss aggregation (ELT to OEP/AEP curves).

## Licence

MIT.
