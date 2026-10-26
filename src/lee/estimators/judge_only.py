"""Judge-only prevalence: ignore human labels (diagnostic / biased)."""

from __future__ import annotations

import numpy as np
from scipy import stats

from lee.estimators.types import IntervalEstimate


def judge_only_mean(
    f_all: np.ndarray,
    *,
    alpha: float = 0.05,
) -> IntervalEstimate:
    """Mean of soft scores on the full population, with a naive CLT interval."""
    f = np.asarray(f_all, dtype=float).ravel()
    n = f.size
    if n < 1:
        raise ValueError("judge_only_mean requires N >= 1")
    mean = float(f.mean())
    se = float(f.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return IntervalEstimate(
        estimate=mean,
        lower=mean - z * se,
        upper=mean + z * se,
        alpha=alpha,
        method="judge_only_mean",
        lam=None,
    )


def judge_only_thresholded(
    f_all: np.ndarray,
    *,
    threshold: float = 0.5,
    alpha: float = 0.05,
) -> IntervalEstimate:
    """Mean of hard labels 1{f >= threshold}, with a naive CLT interval."""
    f = np.asarray(f_all, dtype=float).ravel()
    hard = (f >= threshold).astype(float)
    est = judge_only_mean(hard, alpha=alpha)
    return IntervalEstimate(
        estimate=est.estimate,
        lower=est.lower,
        upper=est.upper,
        alpha=alpha,
        method=f"judge_only_threshold_{threshold:g}",
        lam=None,
    )
