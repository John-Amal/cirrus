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

## 2026-09-21 — Phase 1: data pipeline and augmentation

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

## 2026-09-23 — Phase 2: pretraining

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

## 2026-09-26 — Phase 3: objectives

- CSGD chosen for precipitation: it folds the point mass at zero into one
  distribution, so no separate rain/no-rain classifier with its own loss and
  calibration is needed.
- Blocker that shaped the design: `torch.special.gammainc` has **no
  derivative with respect to its shape parameter**, so a CSGD likelihood
  cannot be backpropagated through. Sample-based CRPS instead —
  `Gamma.rsample` uses implicit reparameterisation and does carry gradients
  to the shape. Verified the estimator against the closed-form Gaussian CRPS
  (0.7136 vs 0.7143 exact at m=20) and confirmed the biased normalisation
  scores higher, which is why the fair one is the default: the biased version
  rewards narrowing the spread.
- CRPS spread term is O(m²). m=1000 allocated 30 GB in a test; 20–30 samples
  is the useful range.
- Thresholds defined by a fixed exceedance *rate* (2% of steps) rather than a
  fixed quantile, so every cell contributes a comparable number of events for
  the Phase 4 GPD fits. Result: median 3.26 mm/6h, only 1.1% of cells on the
  0.1 mm floor, median 1052 exceedances per cell.
- All four arms predict **millimetres**, not log space, which removes the
  Phase 2 transform bias from the comparison entirely.
- Result: distributional objectives reproduce tail amplitude (p99.9 draw
  ratio 1.00–1.03), deterministic ones do not (0.74–0.82). Cost is 3% of MAE;
  gain is 24% on exceedance Brier.
- **Negative result worth keeping:** aggressive tail weighting (p98) is
  *worse than plain CRPS on every metric*, including the tail-weighted one it
  was trained for. The chaining function silences ~98% of the signal, and the
  CSGD's three parameters couple bulk and tail, so discarding bulk
  observations discards information that was constraining the tail. A milder
  p90 threshold beats both. Non-monotone in threshold severity.
- Three seeds of `crps` and `twcrps_p90` separate cleanly (0.0268 vs
  0.0264–0.0265, no overlap). Seeds vary head init and batch order, not the
  augmentation draw — so this is optimisation variance, not pipeline
  variance. Say which was measured.

## 2026-09-29 — Phase 4: return levels

- Both evaluation paths were loader-bound with num_workers=0: ~4.6 s/batch
  loading vs ~0.3 s compute. The Phase 2 throughput measurement had already
  established this and it was not carried into eval code. Check the loader
  before optimising anything else.
- **The main finding.** With ξ estimated freely, the deterministic arms give
  ξ = −0.134 (L1) and −0.089 (MSE): *negative*, meaning a distribution with a
  finite upper bound. Observations give +0.002, distributional arms +0.055 to
  +0.074. The failure is the wrong tail *shape*, not merely a small scale —
  which is why their deficit worsens with return period (L1: 0.72 → 0.66 →
  0.62) while the distributional arms stay flat.
- Fixing ξ would have hidden this entirely. Running the free variant as well
  as the constrained ones was the decision that made the finding visible.
- Shape stationarity tested rather than assumed: ξ fitted independently on
  1979–2014 and 2017–2022 agrees for 80% of sampled cells, median |Δξ| 0.041.
- The pooled global ξ (+0.288) is **biased heavy** against a per-cell median
  of +0.023: pooling excesses across cells with heterogeneous scales creates
  a mixture with a heavier tail than any component. Sensitivity check only.
- Persistence doubles as an end-to-end validation of the apparatus: it pushes
  observations through sampler, grouping, fitting and return level and
  returns 1.00/1.00/1.00 with ξ +0.002. It also shows reproducing the tail is
  trivial without skill, since it scores worst but one on CRPS and Brier.
- Test period's tail is ~4% heavier than training. Six-year windows *within*
  the training record span 0.97–1.03, so 1.04 is marginally outside —
  suggestive of a trend, not separable from internal variability at this
  sample size. Calibration under the null showed the free-shape variant is
  biased low (median 0.953–0.974) and cannot resolve a 4% effect; the
  fixed-shape variant is unbiased ([0.994, 1.003]) and can.
- Climatology under-predicts partly for that reason and partly because a
  resampling climatology cannot exceed the training-period maximum: that
  ceiling sits below the observed 20-year level in 39% of cells. Extending
  quantile levels from 1−3e−5 to 1−1e−6 moved it only 42% → 39%, as it must.
- Sampling bug caught by checking against a known distribution rather than
  checking it ran: the climatology quantile levels are deliberately uneven,
  so drawing a uniform *index* oversampled the tail and inflated the mean
  3.6-fold. Sampling has to happen in probability space.

## 2026-10-03 — Phase 5: export, serving, deployment

- Export: TorchScript reproduces PyTorch **bit-identically** (max diff
  0.00e+00) at 9.9 ms/batch on CPU. ONNX exports cleanly once `onnxscript`
  is installed — recent torch routes `torch.onnx.export` through it, and
  neither `onnx` nor `onnxruntime` pulls it in.
- Dynamic int8 quantisation measured and **rejected**: max diff 2.1e-01
  (against a shape parameter of order 0.02) and *slower* than float32
  (11.1 ms vs 9.9). At 5M parameters over 128 tokens the matmuls are too
  small for int8 to repay its overhead. Also needs a backend compiled into
  the build; some macOS wheels have none, which surfaced as `NoQEngine`
  deep inside the conversion. Now detected and reported up front.
- `torch.jit.trace` and `torch.jit.freeze` are deprecated in favour of
  `torch.export`. TorchScript still has the widest deployment support;
  migration is future work.
- Hugging Face Spaces now requires a paid plan for Docker SDK Spaces, so
  deployment moved to Render's free tier: 512 MB RAM, spin-down after 15
  minutes, 30–60 s cold start. PyTorch alone is ~300 MB resident, so the
  service was made runtime-agnostic and deploys with ONNX: image 1.5 GB →
  under 300 MB, resident well under 150 MB.

### Deployment failures, all outside the model code

- `.gitignore` rule `serve/` matched at any depth and excluded
  `src/cirrus/serve/` from git — the **second** occurrence of this pattern
  after `data/` excluded `src/cirrus/data/` in Phase 1. Symptom was
  `ModuleNotFoundError` in deployment while local tests passed. Anchor
  directory ignores with a leading slash; verify with `git ls-files`.
- Docker's layer cache reused a `pip install git+https://...` layer built
  from an older commit, because the instruction text is identical between
  builds regardless of what the remote contains. Cleared the build cache.
- The "torch-free" image still failed on `import xarray`: `api.py` imports
  `IngestSpec` from `ingest.py`, which imports xarray at module level for
  download machinery the service never uses. Installing xarray was the
  expedient fix; the correct one is to read the channel order from
  `bundle.json` and drop the dependency. Mixing a spec dataclass and I/O
  code in one module means importing the former costs the latter's
  dependencies.
- CI: the `serve` extra was added as a *new step at the end* of the
  workflow, after `mypy`. Steps run in order, so the install happened after
  the check that needed it. Position in a pipeline is part of the
  configuration.

Theme, here and across the CI failures earlier in the project: the
environment that builds and the environment that develops diverge in ways
local testing cannot reveal. Local green is not green.
