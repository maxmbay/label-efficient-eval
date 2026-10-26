"""Unit tests for CUPAC / PPI identity and A/A smoke."""

from __future__ import annotations

import numpy as np

from lee.estimators.cupac import (
    cupac_ate,
    cupac_from_features,
    diff_in_means_ate,
    ppi_form_mean_from_cupac,
)
from lee.estimators.ppi import ppi_point_estimate


def test_ppi_form_identity_numerical() -> None:
    rng = np.random.default_rng(0)
    n = 400
    f = rng.uniform(0, 1, size=n)
    y = (rng.random(n) < np.clip(0.2 + 0.5 * f, 0, 1)).astype(float)
    cupac_like, ppi_like, lam = ppi_form_mean_from_cupac(y, f, lam=None)
    assert abs(cupac_like - ppi_like) <= 1e-12
    # Cross-check against ppi_point_estimate on the same split.
    n_lab = n // 2
    assert abs(
        ppi_like
        - ppi_point_estimate(y[:n_lab], f[:n_lab], f[n_lab:], lam=lam)
    ) <= 1e-12


def test_cupac_reduces_var_when_f_predictive() -> None:
    rng = np.random.default_rng(1)
    n = 5000
    x = rng.normal(size=(n, 3))
    # Outcome depends on x; treatment is randomized with a small effect.
    logit = x[:, 0] - 0.5 * x[:, 1]
    p = 1 / (1 + np.exp(-logit))
    t = rng.binomial(1, 0.5, size=n).astype(float)
    y = (rng.random(n) < np.clip(p + 0.05 * t, 0, 1)).astype(float)
    # Use true p as oracle f (centered CUPAC).
    dim = diff_in_means_ate(y, t)
    cupac = cupac_ate(y, t, p, lam=1.0)
    assert cupac.var < dim.var


def test_aa_dim_near_nominal() -> None:
    rng = np.random.default_rng(2)
    n = 4000
    y = rng.binomial(1, 0.05, size=n).astype(float)
    rejects = 0
    R = 100
    for r in range(R):
        rr = np.random.default_rng(3000 + r)
        t = rr.integers(0, 2, size=n).astype(float)
        est = diff_in_means_ate(y, t, alpha=0.05)
        rejects += int(est.rejects_zero())
    # Loose band around 5%.
    assert 0.0 <= rejects / R <= 0.15


def test_cupac_from_features_runs() -> None:
    rng = np.random.default_rng(3)
    n = 800
    x = rng.normal(size=(n, 4))
    t = rng.integers(0, 2, size=n).astype(float)
    y = (rng.random(n) < 0.1 + 0.02 * t).astype(float)
    ate, f, lam = cupac_from_features(x, y, t, n_folds=4, seed=0)
    assert f.shape == (n,)
    assert np.isfinite(ate.estimate)
    assert 0.0 <= lam <= 1.0
