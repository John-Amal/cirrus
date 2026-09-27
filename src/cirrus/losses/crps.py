"""Proper scoring rules, estimated from samples.

CRPS has a closed form for a handful of distributions and not for the
censored shifted gamma we need. More to the point, the closed form would
require the incomplete gamma function, whose derivative with respect to the
shape parameter PyTorch does not implement -- so it could not be trained
through in any case.

The way around both problems is the sample estimator. ``Gamma.rsample`` uses
implicit reparameterisation and *does* carry gradients to the shape, so a
score computed from draws is differentiable end to end.

The estimator (Gneiting and Raftery 2007) is

    CRPS(F, y) = E|X - y| - 0.5 E|X - X'|

for independent draws X, X' from F. The first term rewards being close to
the observation; the second penalises being over-dispersed. A distribution
concentrated on the wrong value and a wide vague one both score badly, which
is what makes CRPS proper and what makes it unlike squared error: it cannot
be gamed by predicting the conditional mean.
"""

from __future__ import annotations

import torch

EPS = 1e-12


def crps_sample(
    samples: torch.Tensor, target: torch.Tensor, fair: bool = True
) -> torch.Tensor:
    """CRPS estimated from samples, elementwise.

    Args:
        samples: ``(..., m)`` draws from the predictive distribution.
        target: ``(...)`` observations.
        fair: Use the unbiased ``m(m-1)`` normalisation for the spread term.
            With the biased ``m**2`` version, a model can lower its score by
            reducing its spread, so it is nudged toward over-confidence --
            precisely the failure this project is about.

    Returns:
        ``(...)`` scores, in the units of the target. Lower is better.

    """
    if samples.shape[:-1] != target.shape:
        raise ValueError(
            f"samples {tuple(samples.shape)} and target {tuple(target.shape)} "
            "must agree on all but the sample dimension"
        )
    n_samples = samples.shape[-1]
    if fair and n_samples < 2:
        raise ValueError("the fair estimator needs at least two samples")

    accuracy = (samples - target.unsqueeze(-1)).abs().mean(dim=-1)
    pairwise = (samples.unsqueeze(-1) - samples.unsqueeze(-2)).abs()
    denominator = n_samples * (n_samples - 1) if fair else n_samples * n_samples
    spread = pairwise.sum(dim=(-2, -1)) / denominator
    return accuracy - 0.5 * spread


def threshold_weighted_crps(
    samples: torch.Tensor,
    target: torch.Tensor,
    threshold: torch.Tensor,
    fair: bool = True,
) -> torch.Tensor:
    """CRPS concentrated above a threshold, via a chaining function.

    Applying ``v(x) = max(x, u)`` to both samples and observation before
    scoring leaves the rule proper while making it blind to differences that
    occur entirely below ``u``. Getting a dry day slightly wrong stops
    mattering; getting the magnitude of a heavy event wrong still does.

    This is the tail-aware objective the project tests. Unlike bolting a
    second distribution onto the model, it introduces no weighting between
    competing likelihoods and no extra parameters.

    Args:
        samples: ``(..., m)`` draws.
        target: ``(...)`` observations.
        threshold: broadcastable to ``target``; per grid cell in practice.
        fair: Use the unbiased spread normalisation, as in :func:`crps_sample`.

    """
    # Always add the sample axis. The threshold may be (H, W) while the
    # target is (B, H, W): appending the axis lets both broadcast against
    # (B, H, W, m), whereas a bare (H, W) would misalign against (W, m).
    limit = threshold.unsqueeze(-1)
    return crps_sample(
        torch.maximum(samples, limit),
        torch.maximum(target, threshold),
        fair=fair,
    )


def mean_score(
    score: torch.Tensor, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Average an elementwise score, optionally with latitude weights."""
    if weights is None:
        return score.mean()
    broadcast = weights.reshape((1,) * (score.ndim - weights.ndim) + weights.shape)
    return (score * broadcast).sum() / broadcast.expand_as(score).sum().clamp(min=EPS)
