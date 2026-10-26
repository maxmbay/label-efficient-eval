"""CUPAC (ML outcome adjustment) and its PPI-form identity for means / ATE.

CUPAC, not CUPED: the control variate is a cross-fit ML prediction of the
outcome, not a pre-period metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from lee.estimators.ppi import optimal_lam, ppi_point_estimate
from lee.estimators.types import IntervalEstimate


@dataclass(frozen=True)
class AteEstimate:
    estimate: float
    se: float
    lower: float
    upper: float
    alpha: float
    method: str
    var: float

    def rejects_zero(self) -> bool:
        return (self.lower > 0.0) or (self.upper < 0.0)


def diff_in_means_ate(
    y: np.ndarray,
    t: np.ndarray,
    *,
    alpha: float = 0.05,
) -> AteEstimate:
    """Simple difference in means ATE under randomization."""
    y = np.asarray(y, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    if y.size != t.size:
        raise ValueError("y and t length mismatch")
    y1 = y[t == 1]
    y0 = y[t == 0]
    if y1.size < 2 or y0.size < 2:
        raise ValueError("need at least 2 observations per arm")
    m1, m0 = float(y1.mean()), float(y0.mean())
    v1 = float(y1.var(ddof=1)) / y1.size
    v0 = float(y0.var(ddof=1)) / y0.size
    se = float(np.sqrt(v1 + v0))
    est = m1 - m0
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return AteEstimate(
        estimate=est,
        se=se,
        lower=est - z * se,
        upper=est + z * se,
        alpha=alpha,
        method="dim",
        var=v1 + v0,
    )


def crossfit_outcome_scores(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_folds: int = 5,
    seed: int = 20261026,
) -> np.ndarray:
    """Out-of-fold predictions of Y from X via HistGradientBoostingRegressor."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("x must be (n, p) aligned with y")
    n = y.size
    preds = np.zeros(n, dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(x):
        model = HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.08,
            max_iter=120,
            random_state=seed,
        )
        model.fit(x[train_idx], y[train_idx])
        preds[test_idx] = model.predict(x[test_idx])
    return preds


def cupac_adjusted_outcomes(
    y: np.ndarray,
    f: np.ndarray,
    *,
    lam: float | None = None,
) -> tuple[np.ndarray, float]:
    """Return Y - lam * (f - mean(f)) and the lambda used.

    With lam = Cov(Y,f)/Var(f) this is the PPI++ / CUPAC power-tuned form for
    a single mean. Using the centered predictor keeps the population mean of
    the adjustment at zero.
    """
    y = np.asarray(y, dtype=float).ravel()
    f = np.asarray(f, dtype=float).ravel()
    if y.size != f.size:
        raise ValueError("y and f length mismatch")
    f_bar = float(f.mean())
    f_c = f - f_bar
    if lam is None:
        # Match PPI++ optimal lam on a fully-labeled population: unlabeled
        # pool is empty, so use Cov/Var on the labeled sample.
        if float(np.var(f_c, ddof=1)) <= 0:
            lam_used = 0.0
        else:
            lam_used = float(np.cov(y, f, ddof=1)[0, 1] / np.var(f, ddof=1))
            lam_used = float(np.clip(lam_used, 0.0, 1.0))
    else:
        lam_used = float(lam)
    return y - lam_used * f_c, lam_used


def cupac_ate(
    y: np.ndarray,
    t: np.ndarray,
    f: np.ndarray,
    *,
    alpha: float = 0.05,
    lam: float | None = None,
) -> AteEstimate:
    """ATE as difference in means of CUPAC-adjusted outcomes."""
    adj, lam_used = cupac_adjusted_outcomes(y, f, lam=lam)
    base = diff_in_means_ate(adj, t, alpha=alpha)
    return AteEstimate(
        estimate=base.estimate,
        se=base.se,
        lower=base.lower,
        upper=base.upper,
        alpha=alpha,
        method=f"cupac_lam_{lam_used:g}",
        var=base.var,
    )


def cupac_from_features(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    *,
    n_folds: int = 5,
    seed: int = 20261026,
    alpha: float = 0.05,
    lam: float | None = None,
) -> tuple[AteEstimate, np.ndarray, float]:
    """Cross-fit m(x), then CUPAC ATE. Returns (ate, oof_preds, lam)."""
    f = crossfit_outcome_scores(x, y, n_folds=n_folds, seed=seed)
    ate = cupac_ate(y, t, f, alpha=alpha, lam=lam)
    _, lam_used = cupac_adjusted_outcomes(y, f, lam=lam)
    return ate, f, lam_used


def ppi_form_mean_from_cupac(
    y: np.ndarray,
    f: np.ndarray,
    *,
    lam: float | None = None,
) -> tuple[float, float, float]:
    """Show CUPAC mean == PPI mean on a fully labeled sample.

    Returns (cupac_mean, ppi_mean, lam). For a fully labeled sample the
    unlabeled pool is a bootstrap of f itself for the identity check; the
    algebraic identity for the *adjusted mean* is:

        mean(Y - lam*(f - mean(f))) == mean(Y) - lam*(mean(f) - mean(f))
                                    == mean(Y)

    when the same f is used everywhere. The interesting identity is that the
    *estimator with a held-out unlabeled f pool* matches PPI. Here we split
    indices 50/50: labeled vs "unlabeled" predictions, compare CUPAC-style
    rectification to ``ppi_point_estimate``.
    """
    y = np.asarray(y, dtype=float).ravel()
    f = np.asarray(f, dtype=float).ravel()
    n = y.size
    if n < 4:
        raise ValueError("need n >= 4 for the split identity check")
    n_lab = n // 2
    y_l, f_l = y[:n_lab], f[:n_lab]
    f_u = f[n_lab:]
    if lam is None:
        lam_used = optimal_lam(y_l, f_l, f_u, clip=True)
    else:
        lam_used = float(lam)
    # CUPAC / PPI point estimate on the split.
    ppi = ppi_point_estimate(y_l, f_l, f_u, lam=lam_used)
    # Explicit rectification written as mean(Y_l) - lam*(mean(f_l)-mean(f_u)).
    cupac_like = float(y_l.mean() - lam_used * (f_l.mean() - f_u.mean()))
    return cupac_like, float(ppi), lam_used


def mean_interval_classical(
    y: np.ndarray,
    *,
    alpha: float = 0.05,
) -> IntervalEstimate:
    y = np.asarray(y, dtype=float).ravel()
    m = float(y.mean())
    se = float(y.std(ddof=1) / np.sqrt(y.size)) if y.size > 1 else 0.0
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return IntervalEstimate(
        estimate=m,
        lower=m - z * se,
        upper=m + z * se,
        alpha=alpha,
        method="classical_mean",
        lam=0.0,
    )
