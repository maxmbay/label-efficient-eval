"""Unit tests for hedged betting confidence sequences."""

from __future__ import annotations

import numpy as np

from lee.estimators.betting_cs import (
    HedgedBettingCS,
    affine_to_unit,
    first_clear_above,
    ppi_rectifier_bounds,
)


def test_cs_covers_known_mean() -> None:
    rng = np.random.default_rng(0)
    mu = 0.42
    cs = HedgedBettingCS(alpha=0.05, grid=501)
    x = rng.binomial(1, mu, size=400).astype(float)
    for xi in x:
        cs.update(float(xi))
    assert cs.lower <= mu <= cs.upper


def test_running_intersection_monotone() -> None:
    rng = np.random.default_rng(1)
    cs = HedgedBettingCS(alpha=0.1, grid=301)
    lowers, uppers = [], []
    for xi in rng.uniform(0.2, 0.8, size=100):
        s = cs.update(float(xi))
        lowers.append(s.lower)
        uppers.append(s.upper)
    # Running intersection: lower nondecreasing, upper nonincreasing.
    assert all(lowers[i] <= lowers[i + 1] + 1e-12 for i in range(len(lowers) - 1))
    assert all(uppers[i] >= uppers[i + 1] - 1e-12 for i in range(len(uppers) - 1))


def test_clear_above_detects() -> None:
    cs = HedgedBettingCS(alpha=0.05, grid=401)
    states = [cs.update(0.9) for _ in range(80)]
    t = first_clear_above(states, threshold=0.5)
    assert t is not None
    assert states[t - 1].lower > 0.5


def test_rectifier_affine_roundtrip() -> None:
    z_min, z_max = ppi_rectifier_bounds(lam=1.0)
    z = np.array([-0.5, 0.0, 1.0, 1.5])
    u = affine_to_unit(z, z_min=z_min, z_max=z_max)
    assert u.min() >= -1e-12 and u.max() <= 1 + 1e-12
