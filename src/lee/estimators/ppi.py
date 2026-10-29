"""Prediction-powered inference (PPI) and PPI++ for a scalar mean.

Formulas match Angelopoulos et al. (PPI / PPI++) and are cross-checked against
``ppi_py.ppi_mean_ci`` / ``ppi_mean_pointestimate`` in tests.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from lee.estimators.types import IntervalEstimate


def ppi_point_estimate(
    y: np.ndarray,
    yhat: np.ndarray,
    yhat_unlabeled: np.ndarray,
    *,
    lam: float = 1.0,
) -> float:
    """theta = mean(Y - lam * f_lab) + mean(lam * f_unlab)."""
    y = np.asarray(y, dtype=float).ravel()
    yhat = np.asarray(yhat, dtype=float).ravel()
    yhat_u = np.asarray(yhat_unlabeled, dtype=float).ravel()
    return float((y - lam * yhat).mean() + (lam * yhat_u).mean())


def optimal_lam(
    y: np.ndarray,
    yhat: np.ndarray,
    yhat_unlabeled: np.ndarray,
    *,
    clip: bool = True,
) -> float:
    """Power-tuning lambda for a 1-d mean (PPI++), matching ppi_py.

    Uses a preliminary PPI (lam=1) point estimate for centering, then the
    GLM mean formula in ``ppi_py.ppi._calc_lam_glm`` with ``clip=True``.
    """
    y = np.asarray(y, dtype=float).ravel()
    yhat = np.asarray(yhat, dtype=float).ravel()
    yhat_u = np.asarray(yhat_unlabeled, dtype=float).ravel()
    n = y.size
    n_u = yhat_u.size
    if n < 2 or n_u < 1:
        return 0.0

    pointest = ppi_point_estimate(y, yhat, yhat_u, lam=1.0)
    grads = y - pointest
    grads_hat = yhat - pointest
    grads_hat_u = yhat_u - pointest

    grads_cent = grads - grads.mean()
    grad_hat_cent = grads_hat - grads_hat.mean()
    # (1/n) * (g' h + h' g) for scalars = 2 * mean(g_cent * h_cent)
    cov_grads = (2.0 / n) * float(np.dot(grads_cent, grad_hat_cent))

    concat = np.concatenate([grads_hat, grads_hat_u])
    var_grads_hat = float(np.cov(concat, ddof=1)) if concat.size > 1 else 0.0
    denom = 2.0 * (1.0 + n / n_u) * var_grads_hat
    if denom <= 0.0 or not np.isfinite(denom):
        lam = 0.0
    else:
        lam = cov_grads / denom
    if clip:
        lam = float(np.clip(lam, 0.0, 1.0))
    return float(lam)


def ppi_mean_ci(
    y: np.ndarray,
    yhat: np.ndarray,
    yhat_unlabeled: np.ndarray,
    *,
    alpha: float = 0.05,
    lam: float | None = None,
) -> IntervalEstimate:
    """PPI (lam=1) or PPI++ (lam=None -> power-tuned) CI for a scalar mean."""
    y = np.asarray(y, dtype=float).ravel()
    yhat = np.asarray(yhat, dtype=float).ravel()
    yhat_u = np.asarray(yhat_unlabeled, dtype=float).ravel()
    n = y.size
    n_u = yhat_u.size
    if n < 1 or n_u < 1:
        raise ValueError("PPI requires n >= 1 and N_unlabeled >= 1")

    method = "ppi"
    if lam is None:
        lam = optimal_lam(y, yhat, yhat_u, clip=True)
        method = "ppi_pp"
    else:
        lam = float(lam)
        method = "ppi" if abs(lam - 1.0) < 1e-12 else f"ppi_lam_{lam:g}"

    estimate = ppi_point_estimate(y, yhat, yhat_u, lam=lam)

    # Match ppi_py: population std (ddof=0) / sqrt(n)
    imputed_std = float((lam * yhat_u).std(ddof=0) / np.sqrt(n_u))
    rectifier_std = float((y - lam * yhat).std(ddof=0) / np.sqrt(n))
    se = float(np.sqrt(imputed_std**2 + rectifier_std**2))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return IntervalEstimate(
        estimate=estimate,
        lower=estimate - z * se,
        upper=estimate + z * se,
        alpha=alpha,
        method=method,
        lam=lam,
    )


def ppi_ci(
    y: np.ndarray,
    yhat: np.ndarray,
    yhat_unlabeled: np.ndarray,
    *,
    alpha: float = 0.05,
) -> IntervalEstimate:
    """Standard PPI with lambda = 1."""
    return ppi_mean_ci(y, yhat, yhat_unlabeled, alpha=alpha, lam=1.0)


def ppi_pp_ci(
    y: np.ndarray,
    yhat: np.ndarray,
    yhat_unlabeled: np.ndarray,
    *,
    alpha: float = 0.05,
) -> IntervalEstimate:
    """PPI++ with power-tuned lambda."""
    return ppi_mean_ci(y, yhat, yhat_unlabeled, alpha=alpha, lam=None)
