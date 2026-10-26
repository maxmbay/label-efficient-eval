"""Smoke test that the package imports."""

from lee import __version__


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_m3_imports() -> None:
    from lee.estimators import aipw_mean_ci, cupac_ate, HedgedBettingCS
    from lee.sim import m3_allocation, m3_criteo, m3_stopping

    assert callable(aipw_mean_ci)
    assert callable(cupac_ate)
    assert HedgedBettingCS is not None
    assert m3_allocation.SEED_ROOT == 20261026
    assert m3_stopping.TAU == 0.10
    assert m3_criteo.R_AA_FULL == 200
