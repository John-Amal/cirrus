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
