"""Tests for pairwise win-rate PPI++ wrappers."""

from __future__ import annotations

import numpy as np
import pytest

from lee.estimators.pairwise import (
    bradley_terry_strengths,
    pairwise_winrate_ppi,
    pairwise_winrate_ppi_pp,
)
from lee.models.judge_error import (
    N_OUTCOMES,
    OUTCOME_CLASSES,
    JudgeErrorData,
    build_model,
    fit_nuts,
)


def test_pairwise_ppi_pp_synthetic() -> None:
    rng = np.random.default_rng(0)
    n, n_u = 80, 200
    # Correlated judge: f ~= Y with noise
    y = rng.binomial(1, 0.65, size=n).astype(float)
    yhat = np.clip(y + rng.normal(0, 0.2, size=n), 0, 1)
    yhat_u = np.clip(
        rng.binomial(1, 0.65, size=n_u).astype(float) + rng.normal(0, 0.2, size=n_u),
        0,
        1,
    )
    est = pairwise_winrate_ppi_pp(y, yhat, yhat_u, alpha=0.05)
    assert est.method == "ppi_pp"
    assert est.lam is not None
    assert 0.0 <= est.lam <= 1.0
    assert est.lower < est.estimate < est.upper

    est1 = pairwise_winrate_ppi(y, yhat, yhat_u, alpha=0.05)
    assert est1.method == "ppi"
    assert est1.lam == pytest.approx(1.0)


def test_bt_pairwise_stub_and_judge_error_constants() -> None:
    with pytest.raises(NotImplementedError):
        bradley_terry_strengths(
            np.array(["model_a"]),
            np.array(["a"]),
            np.array(["b"]),
        )
    assert OUTCOME_CLASSES == ("model_a", "model_b", "tie")
    assert N_OUTCOMES == 3
    # build_model / fit_nuts are implemented; smoke coverage lives in
    # scripts/run_m2_judge_error.py (optional jax extra).
    _ = JudgeErrorData
    _ = build_model
    _ = fit_nuts
