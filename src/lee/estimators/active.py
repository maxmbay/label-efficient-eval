"""Active statistical inference (uncertainty-weighted labeling + AIPW).

Follows Zrnic and Candes, Active Statistical Inference (ICML 2024):
theta_hat = mean_i [ f_i + xi_i / pi_i * (Y_i - f_i) ].
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from lee.estimators.types import IntervalEstimate

PI_MIN_DEFAULT = 0.01
EPS_U = 1e-8


def uncertainty_scores(f: np.ndarray) -> np.ndarray:
    """Bernoulli variance proxy f*(1-f) for scores in [0, 1]."""
    f = np.asarray(f, dtype=float).ravel()
    return f * (1.0 - f)


def make_propensity(
    f: np.ndarray,
    *,
    n_budget: int,
    pi_min: float = PI_MIN_DEFAULT,
    eps_u: float = EPS_U,
) -> np.ndarray:
    """Uncertainty-weighted propensities with mean approximately n_budget / N.

    Clips to [pi_min, 1], then rescales toward the target mean when feasible.
    """
    f = np.asarray(f, dtype=float).ravel()
    n = f.size
    if n < 1:
        raise ValueError("make_propensity requires N >= 1")
    if not (1 <= n_budget <= n):
        raise ValueError(f"n_budget must be in [1, N]={n}, got {n_budget}")

    target = n_budget / n
    u = np.maximum(uncertainty_scores(f), eps_u)
    raw = u / u.mean() * target
    pi = np.clip(raw, pi_min, 1.0)

    # One rescaling step toward the target mean under the clip.
    mean_pi = float(pi.mean())
    if mean_pi > 0:
        scale = target / mean_pi
        pi = np.clip(pi * scale, pi_min, 1.0)
        # If still off because of the floor/ceiling, leave as-is and document
        # via the realized mean (caller records mean propensity).
    return pi.astype(float)


def uniform_propensity(n: int, *, n_budget: int) -> np.ndarray:
    if n < 1:
        raise ValueError("uniform_propensity requires N >= 1")
    if not (1 <= n_budget <= n):
        raise ValueError(f"n_budget must be in [1, N]={n}, got {n_budget}")
    return np.full(n, n_budget / n, dtype=float)


def sample_labels(pi: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """Independent Bernoulli draws; returns boolean mask."""
    pi = np.asarray(pi, dtype=float).ravel()
    return rng.random(pi.size) < pi


def aipw_mean_ci(
    y: np.ndarray,
    f: np.ndarray,
    xi: np.ndarray,
    pi: np.ndarray,
    *,
    alpha: float = 0.05,
    method: str = "aipw",
) -> IntervalEstimate:
    """AIPW / active-inference mean with Wald interval on the summands."""
    y = np.asarray(y, dtype=float).ravel()
    f = np.asarray(f, dtype=float).ravel()
    xi = np.asarray(xi, dtype=float).ravel()
    pi = np.asarray(pi, dtype=float).ravel()
    n = y.size
    if not (f.size == xi.size == pi.size == n):
        raise ValueError("y, f, xi, pi must have the same length")
    if np.any(pi <= 0) or np.any(pi > 1):
        raise ValueError("pi must lie in (0, 1]")

    psi = f + (xi / pi) * (y - f)
    estimate = float(psi.mean())
    if n == 1:
        se = 0.0
    else:
        se = float(psi.std(ddof=1) / np.sqrt(n))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return IntervalEstimate(
        estimate=estimate,
        lower=estimate - z * se,
        upper=estimate + z * se,
        alpha=alpha,
        method=method,
        lam=1.0,
    )
