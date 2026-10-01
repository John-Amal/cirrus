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

## Baselines

Two references, chosen because they fail in opposite ways.

| | 20-year return level, model / observed | fitted ξ | MAE (mm) | CRPS | Brier |
| --- | --- | --- | --- | --- | --- |
| persistence | **1.00** | +0.002 | 0.399 | 0.399 | 0.0241 |
| climatology | 0.96 | −0.017 | 0.695 | 0.448 | 0.0230 |
| best trained arm | 1.02–1.09 | +0.055 | 0.314 | 0.224 | 0.0136 |
| *observations* | *1.00* | *+0.002* | — | — | — |

**Persistence doubles as an end-to-end check of the analysis.** It carries the
last observed field forward, so it feeds observations through the entire
apparatus — sampler, exceedance extraction, per-cell GPD fit, return level —
and emerges at 1.00 / 1.00 / 1.00 with ξ = +0.002 against an observed
+0.002. Any error in the sampling, grouping, rate calculation or fitting
would have shown up as a deviation. It did not.

**It also disposes of a tempting misreading.** Persistence reproduces the
observed tail perfectly at every quantile including the maximum, while
scoring worse than every trained arm on CRPS and Brier. Reproducing the tail
is therefore trivially achievable with no forecasting skill whatsoever. The
distributional arms' tail performance means something only because they also
win decisively on the probabilistic scores.

**Climatology is the no-skill reference**, drawing from what each cell does in
each month historically and ignoring the input entirely. Every trained arm
beats it comfortably on every metric — the first direct evidence that the
models learned something about the atmosphere rather than about geography and
season. Its MAE of 0.695 against 0.314 for the best arm is the margin.

One detail worth noting: at a six-hour lead, persistence beats climatology on
MAE by a wide margin (0.399 against 0.695) but is no better at predicting
exceedance *occurrence* (Brier 0.0241 against 0.0230). The two baselines fail
differently, which is why both are reported.

## Is the test period's tail heavier?

Climatology is built from 1979–2014 and scored on 2017–2022, so it
under-predicts if the test period is genuinely heavier. Comparing observed
return levels between the periods, through the identical fit, gives a ratio
of **1.04** at every return period (fixed shape).

Whether that is a trend or interannual variability was tested directly: the
training record was cut into six consecutive six-year windows — the same
length as the test period — and each compared against the full-record fit.
None can differ from that reference for a forced reason, so their spread is
what variability alone produces in a window this short. **They span
0.97–1.03.**

The test period's 1.04 sits just outside that range. Suggestive of a trend,
but one percentage point beyond a range estimated from six windows, and the
windows are spatially averaged over correlated cells, so the true spread is
wider than the six samples show. The honest statement is that the test
period's tail is about 4% heavier, marginally above what six-year windows
span within the training record, and **not separable from internal
variability at this sample size**.

This does not affect the comparison between arms. A shift in the reference
moves every arm equally: the deterministic arms remain at 0.62–0.80 and the
distributional arms at 0.99–1.09 regardless.

Climatology's 0.96 has a second cause as well. A resampling climatology
cannot produce a value larger than the largest ever observed in that cell and
month, and **the training-period maximum lies below the observed 20-year
return level in 39% of cells**. Extending the stored quantile levels from
1 − 3×10⁻⁵ to 1 − 10⁻⁶ moved this only from 42% to 39%, as it must: finer
levels resolve the approach to the sample maximum, they do not lift it. Both
mechanisms contribute and neither dominates.

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

- **Nothing about operational skill.** Persistence and climatology are the
  only baselines; no NWP forecast has been compared against. Beating
  climatology shows the models learned something about the atmosphere, not
  that they are competitive with an operational system.
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
cirrus climatology
./scripts/run_finetune_arms.sh
./scripts/run_seed_sweep.sh
cirrus compare
cirrus returnlevels
```
