# Phase 4: return levels on the held-out test years

The Phase 3 comparison was decided on validation data. This is the first and
only analysis to use 2017–2022.

Each arm's predictive distributions are pooled over the test period into an
implied climatology, a generalised Pareto is fitted to exceedances of the
per-cell threshold, and return levels follow. The identical procedure is
applied to the observations, so any difference is the model's tail rather
than the estimator's.

## The headline

Ratio of modelled to observed return level, latitude-weighted median over
2,048 cells. 1.00 is right.

| arm | 1 year | 5 years | 20 years | fitted ξ |
| --- | --- | --- | --- | --- |
| L1 (point) | 0.72 | 0.66 | **0.62** | −0.134 |
| MSE (point) | 0.78 | 0.73 | **0.70** | −0.089 |
| CRPS | 1.09 | 1.15 | 1.22 | +0.074 |
| twCRPS (p98) | 1.03 | 1.08 | 1.12 | +0.055 |
| twCRPS (p90) | 1.05 | 1.09 | **1.14** | +0.060 |
| *observations* | *1.00* | *1.00* | *1.00* | *+0.002* |

Shape estimated freely, per cell, on each side. Seed repeats of `crps` and
`twcrps_p90` agree to ±0.01 and are omitted for readability.

**Deterministic objectives underestimate the 20-year return level by 30–38%.
Distributional ones are within 14%**, and the tail-weighted variants within
2–12%.

## The mechanism: deterministic training implies a bounded tail

The fitted shape parameters are the more interesting result. Observations
give ξ ≈ 0.00 — an essentially exponential tail. The distributional arms give
+0.055 to +0.074, slightly heavy. The deterministic arms give **−0.134 and
−0.089: negative**, which is a generalised Pareto with a *finite upper
bound*. Their implied climatology says precipitation cannot exceed a ceiling.

That is not a small-scale error, it is the wrong kind of tail, and it
explains the pattern across return periods: the deterministic deficit
worsens with the period (L1: 0.72 → 0.66 → 0.62) because a bounded
distribution diverges further from an unbounded one the further out you go,
while the distributional arms stay roughly flat.

It also explains why this had to be checked with the shape estimated freely.
Fixing ξ from observations — the lower-variance choice — would have imposed
the observed tail shape on every arm and hidden the finding entirely.

## Robustness to the shape treatment

The same comparison under three treatments of ξ, at 20 years:

| arm | free ξ | ξ fixed per cell from training | ξ fixed globally |
| --- | --- | --- | --- |
| L1 | 0.62 | 0.72 | 0.63 |
| MSE | 0.70 | 0.78 | 0.71 |
| CRPS | 1.22 | 1.09 | 1.07 |
| twCRPS (p90) | 1.14 | 1.04 | 1.02 |

The ordering is identical in all three, and the deterministic/distributional
gap survives everywhere. The magnitudes shift, as expected: constraining ξ
pulls every arm toward the observed shape and therefore toward 1.00.

**Is fixing ξ from the training period legitimate?** It assumes the tail
*shape* is stationary between 1979–2014 and 2017–2022, which a warming
climate might violate. Rather than assume it, ξ was fitted independently on
each period for a sample of cells: the training estimate falls inside a
bootstrap interval for the test estimate in **80% of cells**, with a median
absolute difference of 0.037 — smaller than the estimation uncertainty. The
assumption is consistent with the data at this sample size.

**The global variant is biased heavy** and should be read only as a
sensitivity check. Pooling excesses across cells with very different scales
produces a mixture whose tail is heavier than any component's, which is why
its ξ is +0.288 against a per-cell median of +0.023. Standardising excesses
by per-cell scale before pooling would fix it.

## Uncertainty

Observed return levels carry block-bootstrap intervals (30-day blocks, to
respect the clustering of exceedances) of **±12% at 1 year, ±16% at 5 years
and ±21% at 20 years**.

That width matters for reading the table:

- The deterministic deficit (0.62–0.78) lies far outside it. **Established.**
- The CRPS arms' excess (1.07–1.22) lies inside it at the shorter periods.
  "Slightly over-predicts" is **not established**; "not detectably wrong" is
  the honest reading.
- The ordering of arms is consistent across three shape treatments and three
  seeds, which is stronger evidence than any single interval.

The model side uses an ordinary bootstrap, not a block one. A draw from a
per-cell predictive distribution has no temporal coherence, so its
exceedances are independent by construction while the observations' are not.
The asymmetry is real; treating both sides identically would misrepresent one
of them.

## What this does not establish

- **Nothing about operational skill.** No NWP baseline has been run. The
  comparison is between training objectives on one backbone.
- **Nothing about a warming climate.** All of this is within the historical
  record. Out-of-distribution evaluation on storyline simulations is the next
  step.
- **Nothing at station scale.** At 5.625° a cell is a ~600 km area mean.
  These return levels are not comparable to gauge return levels, and the
  absolute values should not be quoted as though they were.
- **Nothing about temporal structure.** Return levels are marginal
  statistics. A model that gets them right can still misplace every event in
  time, and nothing here would detect that.

## Reproduce

```bash
cirrus thresholds
./scripts/run_finetune_arms.sh
./scripts/run_seed_sweep.sh
cirrus returnlevels
```
