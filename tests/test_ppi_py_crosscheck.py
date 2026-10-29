"""Cross-check in-repo PPI / PPI++ against ppi_py to numerical tolerance."""

from __future__ import annotations

import numpy as np
import pytest
from ppi_py import ppi_mean_ci as ref_ci
from ppi_py import ppi_mean_pointestimate as ref_point

from lee.estimators.ppi import optimal_lam, ppi_mean_ci, ppi_point_estimate


def _as_float(x) -> float:
    return float(np.asarray(x).reshape(-1)[0])


def _make_data(seed: int = 7):
    rng = np.random.default_rng(seed)
    n, N = 120, 900
    y = rng.binomial(1, 0.35, size=n).astype(float)
    yhat = np.clip(0.6 * y + 0.4 * rng.random(n), 0, 1)
    yhat_u = np.clip(0.6 * rng.binomial(1, 0.35, size=N) + 0.4 * rng.random(N), 0, 1)
    return y, yhat, yhat_u


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_point_estimate_matches_ppi_py(lam: float) -> None:
    y, yhat, yhat_u = _make_data()
    ours = ppi_point_estimate(y, yhat, yhat_u, lam=lam)
    theirs = _as_float(ref_point(y, yhat, yhat_u, lam=lam))
    assert ours == pytest.approx(theirs, abs=1e-10)


def test_ppi_ci_lam1_matches_ppi_py() -> None:
    y, yhat, yhat_u = _make_data(8)
    ours = ppi_mean_ci(y, yhat, yhat_u, alpha=0.05, lam=1.0)
    lo, hi = ref_ci(y, yhat, yhat_u, alpha=0.05, lam=1.0)
    assert ours.estimate == pytest.approx(
        _as_float(ref_point(y, yhat, yhat_u, lam=1.0)), abs=1e-10
    )
    assert ours.lower == pytest.approx(_as_float(lo), abs=1e-8)
    assert ours.upper == pytest.approx(_as_float(hi), abs=1e-8)


def test_power_tuned_lam_matches_ppi_py() -> None:
    y, yhat, yhat_u = _make_data(9)
    ours_lam = optimal_lam(y, yhat, yhat_u, clip=True)
    # Recover lambda from reference by comparing point estimates / CI path:
    # ppi_py with lam=None returns CI; also call pointestimate with lam=None
    # which computes the same clipped lambda internally.
    # We compare CI endpoints under power tuning.
    ours = ppi_mean_ci(y, yhat, yhat_u, alpha=0.1, lam=None)
    lo, hi = ref_ci(y, yhat, yhat_u, alpha=0.1, lam=None)
    assert 0.0 <= ours_lam <= 1.0
    assert ours.lam == pytest.approx(ours_lam, abs=1e-12)
    assert ours.lower == pytest.approx(float(np.asarray(lo).ravel()[0]), abs=1e-6)
    assert ours.upper == pytest.approx(float(np.asarray(hi).ravel()[0]), abs=1e-6)
