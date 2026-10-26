"""Unit tests for active AIPW allocation."""

from __future__ import annotations

import numpy as np

from lee.estimators.active import (
    aipw_mean_ci,
    make_propensity,
    sample_labels,
    uniform_propensity,
)


def test_uniform_propensity_mean() -> None:
    pi = uniform_propensity(1000, n_budget=100)
    assert abs(float(pi.mean()) - 0.1) < 1e-12


def test_active_propensity_near_budget() -> None:
    rng = np.random.default_rng(0)
    f = rng.uniform(0, 1, size=2000)
    pi = make_propensity(f, n_budget=200)
    assert pi.min() >= 0.01 - 1e-12
    assert pi.max() <= 1.0 + 1e-12
    # Mean should be close to target; clip can cause small bias.
    assert abs(float(pi.mean()) - 0.1) < 0.02


def test_aipw_unbiased_under_uniform() -> None:
    """Monte Carlo: uniform AIPW mean is close to true mean."""
    rng = np.random.default_rng(1)
    n = 3000
    truth = 0.35
    y = rng.binomial(1, truth, size=n).astype(float)
    f = np.clip(y + rng.normal(0, 0.2, size=n), 0, 1)
    estimates = []
    for r in range(200):
        rr = np.random.default_rng(1000 + r)
        pi = uniform_propensity(n, n_budget=300)
        xi = sample_labels(pi, rng=rr).astype(float)
        est = aipw_mean_ci(y, f, xi, pi, alpha=0.05)
        estimates.append(est.estimate)
    assert abs(float(np.mean(estimates)) - truth) < 0.02


def test_aipw_covers_truth_often() -> None:
    rng = np.random.default_rng(2)
    n = 2000
    truth = 0.4
    y = rng.binomial(1, truth, size=n).astype(float)
    f = np.clip(0.5 * y + 0.25 + rng.normal(0, 0.05, size=n), 0, 1)
    covered = 0
    R = 100
    for r in range(R):
        rr = np.random.default_rng(2000 + r)
        pi = make_propensity(f, n_budget=250)
        xi = sample_labels(pi, rng=rr).astype(float)
        est = aipw_mean_ci(y, f, xi, pi, alpha=0.05)
        covered += int(est.covers(truth))
    # Loose check: not catastrophically miscalibrated.
    assert covered / R > 0.7
