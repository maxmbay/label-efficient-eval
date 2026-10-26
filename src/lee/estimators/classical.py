"""Classical win-rate / mean intervals from human labels only."""

from __future__ import annotations

import numpy as np
from scipy import stats

from lee.estimators.types import IntervalEstimate


def classical_clt(
    y: np.ndarray,
    *,
    alpha: float = 0.05,
) -> IntervalEstimate:
    """Normal (CLT) interval for a mean / Bernoulli win rate (iid)."""
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
    """Wilson score interval for a Bernoulli win rate."""
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


def cluster_robust_se_mean(
    y: np.ndarray,
    cluster_ids: np.ndarray,
) -> float:
    """Cluster-robust SE for a sample mean (CR1 / Liang-Zeger form).

    Uses
    ``Var = (G/(G-1)) * n^{-2} * sum_g (sum_{i in g} (y_i - ybar))^2``.

    When each observation is its own cluster this recovers the usual
    ``std(ddof=1)/sqrt(n)`` up to the ``G/(G-1)`` factor with ``G=n``.
    """
    y = np.asarray(y, dtype=float).ravel()
    cluster_ids = np.asarray(cluster_ids)
    if y.shape[0] != cluster_ids.shape[0]:
        raise ValueError("y and cluster_ids must have the same length")
    n = y.size
    if n < 1:
        raise ValueError("empty sample")
    if n == 1:
        return 0.0

    ybar = float(y.mean())
    # Sum residuals within each cluster
    # pandas-free: group via sorting
    order = np.argsort(cluster_ids, kind="mergesort")
    y_o = y[order]
    c_o = cluster_ids[order]
    # boundaries where cluster id changes
    cuts = np.flatnonzero(c_o[1:] != c_o[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, n]
    g = starts.size
    if g < 2:
        # Single cluster: SE undefined / infinite for cluster-robust; fall back
        # to iid SE so callers still get a finite interval, flagged in method.
        return float(y.std(ddof=1) / np.sqrt(n))

    s_sq = 0.0
    for a, b in zip(starts, ends):
        s = float((y_o[a:b] - ybar).sum())
        s_sq += s * s
    var = (g / (g - 1.0)) * (s_sq / (n * n))
    return float(np.sqrt(var))


def classical_clustered(
    y: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    alpha: float = 0.05,
) -> IntervalEstimate:
    """Mean win rate with SEs clustered by ``cluster_ids`` (e.g. question_id).

    Follows the cluster-robust mean SE used for paired eval metrics
    (Anthropic, "Adding Error Bars to Evals", arXiv 2411.00640): dependence
    within question is absorbed into the sandwich; cross-question variation
    drives the SE. Uses a normal critical value.
    """
    y = np.asarray(y, dtype=float).ravel()
    cluster_ids = np.asarray(cluster_ids)
    if y.size < 1:
        raise ValueError("classical_clustered requires n >= 1")
    mean = float(y.mean())
    se = cluster_robust_se_mean(y, cluster_ids)
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    n_clusters = int(np.unique(cluster_ids).size)
    method = "classical_clustered"
    if n_clusters < 2:
        method = "classical_clustered_fallback_iid"
    return IntervalEstimate(
        estimate=mean,
        lower=mean - z * se,
        upper=mean + z * se,
        alpha=alpha,
        method=method,
        lam=0.0,
    )
