"""Classical prevalence intervals from human labels only."""

from __future__ import annotations

import numpy as np
from scipy import stats

from lee.estimators.types import IntervalEstimate


def classical_clt(
    y: np.ndarray,
    *,
    alpha: float = 0.05,
) -> IntervalEstimate:
    """Normal (CLT) interval for a mean / Bernoulli prevalence."""
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if n < 1:
        raise ValueError("classical_clt requires n >= 1")
    mean = float(y.mean())
    if n == 1:
        se = 0.0
    else:
        se = float(y.std(ddof=1) / np.sqrt(n))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return IntervalEstimate(
        estimate=mean,
        lower=mean - z * se,
        upper=mean + z * se,
        alpha=alpha,
        method="classical_clt",
        lam=0.0,
    )


def classical_wilson(
    y: np.ndarray,
    *,
    alpha: float = 0.05,
) -> IntervalEstimate:
    """Wilson score interval for a Bernoulli prevalence."""
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if n < 1:
        raise ValueError("classical_wilson requires n >= 1")
    k = float(y.sum())
    p = k / n
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return IntervalEstimate(
        estimate=p,
        lower=float(center - half),
        upper=float(center + half),
        alpha=alpha,
        method="classical_wilson",
        lam=0.0,
    )
