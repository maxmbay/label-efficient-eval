"""Synthetic unit tests for classical / judge-only / PPI / PPI++."""

from __future__ import annotations

import numpy as np
import pytest

from lee.estimators import (
    classical_clt,
    classical_wilson,
    judge_only_mean,
    optimal_lam,
    ppi_ci,
    ppi_pp_ci,
)


def test_classical_clt_known_mean() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(0.3, 1.0, size=5000)
    est = classical_clt(y, alpha=0.05)
    assert abs(est.estimate - y.mean()) < 1e-12
    assert est.lower < est.estimate < est.upper
    assert est.covers(0.3) or abs(est.estimate - 0.3) < 0.05


def test_classical_wilson_bounds() -> None:
    y = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1], dtype=float)
    est = classical_wilson(y, alpha=0.05)
    assert 0.0 <= est.lower <= est.estimate <= est.upper <= 1.0
    assert abs(est.estimate - 0.5) < 1e-12


def test_judge_only_biased_when_miscalibrated() -> None:
    y = np.zeros(200)
    f = np.full(200, 0.8)
    jo = judge_only_mean(f, alpha=0.05)
    assert jo.estimate == pytest.approx(0.8)
    assert not jo.covers(0.0)


def test_ppi_unbiased_structure() -> None:
    rng = np.random.default_rng(1)
    n, N = 200, 2000
    # Perfect predictor: Yhat == Y on labeled, so rectifier is 0 and the
    # PPI point estimate equals the unlabeled prediction mean.
    p = 0.25
    y_full = rng.binomial(1, p, size=n + N).astype(float)
    f_full = y_full.copy()
    y, yhat = y_full[:n], f_full[:n]
    yhat_u = f_full[n:]
    est = ppi_ci(y, yhat, yhat_u, alpha=0.05)
    assert est.estimate == pytest.approx(float(yhat_u.mean()), abs=1e-12)
    assert abs(est.estimate - float(y.mean())) < 0.1  # both near p
    assert est.lam == pytest.approx(1.0)


def test_ppi_pp_lambda_in_unit_interval() -> None:
    rng = np.random.default_rng(2)
    n, N = 150, 800
    y = rng.binomial(1, 0.2, size=n).astype(float)
    noise = rng.normal(0, 0.2, size=n)
    yhat = np.clip(y + noise, 0, 1)
    yhat_u = np.clip(rng.binomial(1, 0.2, size=N).astype(float) + rng.normal(0, 0.2, size=N), 0, 1)
    lam = optimal_lam(y, yhat, yhat_u)
    assert 0.0 <= lam <= 1.0
    est = ppi_pp_ci(y, yhat, yhat_u, alpha=0.05)
    assert est.lam is not None
    assert 0.0 <= est.lam <= 1.0
    assert est.lower <= est.upper


def test_ppi_pp_not_worse_than_classical_large_n() -> None:
    """With a useful predictor, PPI++ mean width should beat classical CLT."""
    rng = np.random.default_rng(3)
    n, N = 400, 4000
    y_pop = rng.binomial(1, 0.3, size=n + N).astype(float)
    f_pop = 0.7 * y_pop + 0.3 * rng.random(n + N)
    y, yhat = y_pop[:n], f_pop[:n]
    yhat_u = f_pop[n:]
    clt = classical_clt(y, alpha=0.05)
    pp = ppi_pp_ci(y, yhat, yhat_u, alpha=0.05)
    assert pp.width <= clt.width + 1e-9
