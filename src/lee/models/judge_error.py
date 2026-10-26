"""Hierarchical judge-error model for MT-Bench (NumPyro skeleton).

Dawid-Skene style applied model (not a new method). Outline matches
analysis_plan.md Milestone 2 §7:

- Latent true preference z_i in {A, B, tie} per analysis unit; enumerate or
  marginalize in the likelihood.
- Each judge has a category-specific confusion matrix over {A, B, tie},
  partially pooled toward that judge's overall matrix (logistic-normal or
  hierarchical Dirichlet).
- Position-bias term for LLM judges (presentation order).
- Humans as one rater class with individual deviations from the class mean.
- Inference: NUTS first; SVI only if needed.

This module is a scaffold: model construction and fitting raise
NotImplementedError until the M2 inference pass. Synthetic unit tests cover
constants and the public API surface only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Outcome / latent classes (fixed order for confusion matrices).
OUTCOME_CLASSES: tuple[str, ...] = ("model_a", "model_b", "tie")
N_OUTCOMES: int = len(OUTCOME_CLASSES)


@dataclass(frozen=True)
class JudgeErrorData:
    """Arrays required to fit the hierarchical judge-error model.

    Indices are integer-coded; mappings live in ``meta``.
    """

    # shape (n_obs,): unit, judge, category, observed label in 0..2
    unit_index: Any
    judge_index: Any
    category_index: Any
    label_index: Any
    # 1 if LLM presentation had model_a first in the raw row, else 0
    a_presented_first: Any
    # which observations are human vs LLM judge (1 = human)
    is_human: Any
    meta: dict[str, Any]


@dataclass(frozen=True)
class JudgeErrorFitResult:
    """Placeholder for posterior summaries (coverage eval vs held-out humans)."""

    samples: Any
    diagnostics: dict[str, Any]


def build_model(data: JudgeErrorData):
    """Return a NumPyro model function for ``data``.

    Implementation deferred: requires optional ``jax`` / ``numpyro`` extras.
    """
    raise NotImplementedError(
        "NumPyro hierarchical judge-error model is scaffolded only. "
        "Install lee[jax] and implement build_model before M2 Bayes runs."
    )


def fit_nuts(
    data: JudgeErrorData,
    *,
    num_warmup: int = 500,
    num_samples: int = 500,
    seed: int = 20261026,
) -> JudgeErrorFitResult:
    """Fit with NUTS (stub)."""
    raise NotImplementedError("fit_nuts is not implemented in the M2 scaffold.")


def fit_svi(
    data: JudgeErrorData,
    *,
    num_steps: int = 10_000,
    seed: int = 20261026,
) -> JudgeErrorFitResult:
    """Fit with SVI only if NUTS is too slow (stub)."""
    raise NotImplementedError("fit_svi is not implemented in the M2 scaffold.")


def posterior_winrate_interval(
    result: JudgeErrorFitResult,
    *,
    model_lo: str,
    model_hi: str,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Point estimate and central posterior interval for theta_AB (stub)."""
    raise NotImplementedError(
        "posterior_winrate_interval is not implemented in the M2 scaffold."
    )
