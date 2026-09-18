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
