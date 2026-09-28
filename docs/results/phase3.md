# Phase 3: does the training objective decide whether extremes survive?

Six fine-tuned heads on one frozen backbone, differing only in the objective
they were trained with. Validation split (2015–2016), 40 batches, identical
masking and identical evaluation for every arm.

## The arms

| arm | objective | prediction |
| --- | --- | --- |
| `mse` | squared error | point value |
| `l1` | absolute error | point value |
| `crps` | CRPS | censored shifted gamma |
| `twcrps` | CRPS above the p98 exceedance threshold | censored shifted gamma |
| `twcrps_p90` | CRPS above the p90 exceedance threshold | censored shifted gamma |

`crps` and `twcrps_p90` were each run with three seeds; ranges below are
across those seeds. Every head has ~1.59M trainable parameters (the point
heads have ~8k fewer, under 1%), so capacity is not a confound.

## Skill

| arm | MAE (mm) | CRPS | twCRPS | Brier |
| --- | --- | --- | --- | --- |
| `mse` | 0.3330 | 0.3330 | 0.0343 | 0.0180 |
| `l1` | **0.3135** | 0.3135 | 0.0353 | 0.0176 |
| `crps` | 0.3243–0.3259 | **0.2239–0.2248** | 0.0268 | **0.0136–0.0137** |
| `twcrps` | 0.3961 | 0.2871 | 0.0274 | 0.0144 |
| `twcrps_p90` | 0.3530–0.3569 | 0.2556–0.2577 | **0.0264–0.0265** | 0.0137–0.0138 |

For a point prediction CRPS reduces exactly to absolute error, which is why
the first two rows repeat — a useful internal check rather than a typo.

## Tail amplitude

Ratio of predicted to observed quantiles, pooled over cells and times. 1.00
is right; below 1 means the model cannot produce events as intense as
reality.

| arm | mean field p99.9 | single draw p99.9 |
| --- | --- | --- |
| `mse` | 0.82 | 0.82 |
| `l1` | 0.74 | 0.74 |
| `crps` | 0.85 | **1.02–1.03** |
| `twcrps` | 0.83 | 0.98 |
| `twcrps_p90` | 0.85 | **0.99–1.00** |

The two columns ask different questions. The **mean field** is what you get
by asking each cell for one number; it is smooth by construction, and every
arm sits at 0.74–0.85 there. A **single draw** is one realisation from the
predictive distribution, which keeps the dispersion the model claims. A
deterministic arm has no draw distinct from its mean, so its two columns are
identical — that identity *is* the limitation.

## What the numbers support

**1. Distributional objectives reproduce tail amplitude; deterministic ones
do not.** 0.99–1.03 against 0.74–0.82 at the 99.9th percentile, same
backbone, same data, same schedule. The effect is roughly twenty times the
seed spread. This is the headline result.

**2. The cost is small and the gain is not.** Moving from `l1` to `crps`
costs 0.011 mm of MAE (3%) and improves the exceedance Brier score by 24%
(0.0176 → 0.0136) while fixing the tail deficit.

**3. Tail weighting helps, but only in moderation, and only for magnitude.**
All three `twcrps_p90` runs beat all three `crps` runs on twCRPS
(0.0264–0.0265 against 0.0268, no overlap), and their draw amplitude is
better calibrated (0.99–1.00 against 1.02–1.03, where `crps` is slightly
over-dispersed). But the aggressive p98 threshold is *worse* than plain CRPS
on every metric, so the relationship is non-monotone in threshold severity.

The mechanism: the chaining function silences the score below the threshold,
so a p98 arm trains on ~2% of the signal. Because the CSGD's three
parameters govern bulk and tail jointly, discarding bulk observations
discards information that was constraining the tail through the parametric
link. Moderate weighting keeps enough of both.

**4. Tail weighting does not improve event *discrimination*.** Brier is
0.0137–0.0138 for `twcrps_p90` against 0.0136–0.0137 for `crps` —
marginally worse, consistently. Better magnitude calibration, no better at
saying whether the event happens. For risk applications, where the size of
the loss matters, that trade is worth making; for a warning system, it is
not obviously so.

Against the hypothesis as originally stated — that a tail-aware objective
improves return levels, exceedance frequencies and record intensity —
the answer is **partly: magnitude yes, occurrence no**, and at a measurable
cost in bulk skill.

## What the numbers do not support

- **Any claim about the test years.** 2017–2022 is untouched; the protocol
  is fixed in Phase 4 before it is used once.
- **Any claim about forecast skill relative to operational systems.** No NWP
  baseline has been run. The comparison here is between objectives, not
  against IFS.
- **Variance beyond optimisation noise.** The seeds change head
  initialisation and batch order. They do not change the augmentation draw,
  which is fixed in the config, so the spreads quoted are optimisation
  variance, not pipeline variance.
- **The twCRPS margin, taken alone.** `twcrps_p90` was trained on something
  close to that metric, so its win there is partly circular. The amplitude
  ratio is the less circular evidence, since no arm was trained on it.

## Details worth knowing

**Draw maxima exceed observed maxima** (ratios 1.17–1.37) for the
distributional arms. With ~2.6M sampled values scored against the same
number of observations, a well-dispersed predictive distribution is expected
to produce a larger maximum than any single realisation of reality. This is
correct behaviour, not over-prediction; a ratio of 1.00 there would indicate
under-dispersion.

**Setup.** Frozen backbone (the 5M-parameter MAE encoder from Phase 2), 6
hour lead time from a two-step input window, 5.625° grid, precipitation
targets in millimetres, latitude-weighted losses and metrics, 8 epochs at
~66 minutes each. All arms are scored against the p98 thresholds regardless
of what they trained on, so the yardstick is fixed while the treatment
varies.

**Reproduce.**

```bash
cirrus thresholds
cirrus thresholds --rate 0.10 --out data/stats/thresholds_train_p90.json
./scripts/run_finetune_arms.sh
./scripts/run_seed_sweep.sh
cirrus compare
```
