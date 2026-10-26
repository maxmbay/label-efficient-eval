"""PPI / PPI++ for pairwise win rates (Milestone 2).

Primary estimand: P(model_lo wins | not tie) on majority-collapsed human
labels, with a judge score f on the same units. Algebra matches the scalar
mean PPI++ used in Milestone 1.

Clustered human-only SEs and Bradley-Terry strengths are stubs for a later
pass; see analysis_plan.md Milestone 2.
"""

from __future__ import annotations

import numpy as np

from lee.estimators.ppi import ppi_ci, ppi_mean_ci, ppi_pp_ci
from lee.estimators.types import IntervalEstimate


def pairwise_winrate_ppi(
    y: np.ndarray,
    yhat: np.ndarray,
    yhat_unlabeled: np.ndarray,
    *,
    alpha: float = 0.05,
) -> IntervalEstimate:
    """PPI (lambda=1) CI for a pairwise win rate."""
    return ppi_ci(y, yhat, yhat_unlabeled, alpha=alpha)


def pairwise_winrate_ppi_pp(
    y: np.ndarray,
    yhat: np.ndarray,
    yhat_unlabeled: np.ndarray,
    *,
    alpha: float = 0.05,
) -> IntervalEstimate:
    """PPI++ CI for a pairwise win rate (power-tuned lambda)."""
    return ppi_pp_ci(y, yhat, yhat_unlabeled, alpha=alpha)


def pairwise_winrate_ppi_mean_ci(
    y: np.ndarray,
    yhat: np.ndarray,
    yhat_unlabeled: np.ndarray,
    *,
    alpha: float = 0.05,
    lam: float | None = None,
) -> IntervalEstimate:
    """Generic PPI / PPI++ for win-rate vectors already mapped to [0, 1]."""
    return ppi_mean_ci(y, yhat, yhat_unlabeled, alpha=alpha, lam=lam)


def human_only_winrate_clustered(
    y: np.ndarray,
    *,
    question_ids: np.ndarray,
    alpha: float = 0.05,
) -> IntervalEstimate:
    """Human-only win rate with SEs clustered by question_id (stub).

    Planned: cluster-robust SE respecting paired structure (Anthropic,
    arXiv 2411.00640). Not implemented in this scaffold.
    """
    raise NotImplementedError(
        "Clustered human-only SEs are stubbed; use a simple mean CI for smoke "
        "tests until the M2 inference pass lands."
    )


def bradley_terry_strengths(
    winners: np.ndarray,
    model_a: np.ndarray,
    model_b: np.ndarray,
) -> dict[str, float]:
    """Bradley-Terry strengths across models (stub)."""
    raise NotImplementedError(
        "Bradley-Terry fit is stubbed for Milestone 2 scaffolding."
    )
