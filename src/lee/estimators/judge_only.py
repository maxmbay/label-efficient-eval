"""Judge-only mean (diagnostic; biased under miscalibration)."""

from __future__ import annotations

import numpy as np
from scipy import stats

from lee.estimators.types import IntervalEstimate


def judge_only_mean(
    f: np.ndarray,
    *,
    alpha: float = 0.05,
) -> IntervalEstimate:
    """IID normal interval for the mean of judge scores on all units."""
    f = np.asarray(f, dtype=float).ravel()
    n = f.size
    if n < 1:
        raise ValueError("judge_only_mean requires n >= 1")
    mean = float(f.mean())
    se = 0.0 if n == 1 else float(f.std(ddof=1) / np.sqrt(n))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return IntervalEstimate(
        estimate=mean,
        lower=mean - z * se,
        upper=mean + z * se,
        alpha=alpha,
        method="judge_only",
        lam=None,
    )
