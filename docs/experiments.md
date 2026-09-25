# Experiment log

Dated record of what was tried and what happened — including the failures.
Nothing signals a real researcher faster than a written record of what did
not work.

## 2026-09-18 — Phase 0: scaffolding

- Repo initialised: packaging, linting, type checking, CI, tests.
- `DummyModel` (two convolutions) trains on synthetic tensors whose targets
  are a known linear mixing of the inputs, so the loss is guaranteed to be
  reducible. Purpose is to validate the loop, not to model anything.
- Config is a frozen dataclass loaded from YAML, with unknown keys rejected.
  Hydra deferred to Phase 2, where config composition starts to earn its cost.

Open question: first pretraining resolution — 5.625 deg (fits overnight on the
Mac) or 2.5 deg (stronger headline result, needs a GPU allocation).

- Overfit-one-batch test initially failed: loss plateaued at 25% of its
  starting value. Cause was the test, not the loop — the target was random
  noise, which a weight-sharing conv model cannot memorise regardless of
  training budget. Fixed by using a linear channel mixing as the target.

- CI caught three type errors invisible on the dev machine;
  mypy's python_version pin conflicted with 3.12 numpy stubs.

## 2026-09-21 - Phase 1: Data and augmentation

- Bug: ingest left WeatherBench's (time, lon, lat) order in place while
  claiming canonical (time, lat, lon). Caught by a broadcast error in the
  latitude weighting. Tests missed it because the fixture was tidier than the
  real source. Fixed transpose, made the fixture mirror WB2's dim order, and
  added check_layout() on every store open. Note: a square grid would have
  hidden this entirely.

- Longitude roll must be paired with a clock shift of -k\*24/n_lon hours, or
  the augmentation teaches a false diurnal cycle. Verified with a synthetic
  field defined purely by local solar time, and by checking the test fails
  when the pairing is removed.

- .gitignore rule "data/" matched at any depth and silently excluded
  src/cirrus/data/ from git. Local tests passed for days while CI never had
  the package. Anchor directory ignores with a leading slash, and check
  `git ls-files` for a new package rather than trusting `git status`.

## 2026-09-23 - PHASE 2: Pretraining

- Pretraining throughput, M-series Mac, 5.6deg, 5M params, batch 32:
  0.89 s/step with num_workers=0, 0.29 s/step with num_workers=4.
  Thoroughly dataloader-bound; single-process loading alone is ~4.6 s/batch.
  Epoch ~8 min, so 20 epochs ~2.6 h.

- Pretraining, 20 epochs, val MSE 0.2385 (from ~1.0 at init). Train/val gap
  0.015: underfitting, not overfitting -- capacity/schedule limited.

- Per-variable spread is 30x: geopotential_250 0.021, precipitation 0.618.
  Meridional wind (~0.53) is ~2x harder than zonal (~0.25): no climatology
  to fall back on.

- Tail amplitude over masked patches: ratio 0.38 at p99, 0.30 at p99.9,
  0.30 at the max. Mean ratio 0.76 -- suspect Jensen bias from the log1p
  transform, to be confirmed.

- Tail deficit is transform-amplified, measured on masked val patches:
  log1p space  mean 1.00, p99 0.61, p99.9 0.60
  physical mm  mean 0.76, p99 0.38, p99.9 0.30
  The model is exactly unbiased where it was optimised. A ~1.0 log-unit
  error at p99.9 (MSE ~1.07, unremarkable) is 12.1 mm vs 3.7 mm in physical
  units. The loss is nearly flat exactly where the extremes are -- the
  motivation for the GPD-informed head, now measured rather than assumed.
