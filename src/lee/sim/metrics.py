"""Monte Carlo metrics over repeated labeled subsets."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SimMetrics:
    estimator: str
    judge: str
    estimand: str
    n: int
    R: int
    alpha: float
    coverage: float
    coverage_se: float
    mean_width: float
    bias: float
    rmse: float
    mean_lam: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_runs(
    *,
    estimates: np.ndarray,
    lowers: np.ndarray,
    uppers: np.ndarray,
    truth: float,
    alpha: float,
    estimator: str,
    judge: str,
    estimand: str,
    n: int,
    lams: np.ndarray | None = None,
) -> SimMetrics:
    estimates = np.asarray(estimates, dtype=float)
    lowers = np.asarray(lowers, dtype=float)
    uppers = np.asarray(uppers, dtype=float)
    r = estimates.size
    covered = (lowers <= truth) & (truth <= uppers)
    cov = float(covered.mean())
    cov_se = float(np.sqrt(cov * (1.0 - cov) / r)) if r else float("nan")
    widths = uppers - lowers
    bias = float(estimates.mean() - truth)
    rmse = float(np.sqrt(np.mean((estimates - truth) ** 2)))
    mean_lam = None if lams is None else float(np.mean(lams))
    return SimMetrics(
        estimator=estimator,
        judge=judge,
        estimand=estimand,
        n=n,
        R=r,
        alpha=alpha,
        coverage=cov,
        coverage_se=cov_se,
        mean_width=float(widths.mean()),
        bias=bias,
        rmse=rmse,
        mean_lam=mean_lam,
    )
