"""Latent-variable models (NumPyro)."""

from lee.models.judge_error import (
    N_OUTCOMES,
    OUTCOME_CLASSES,
    JudgeErrorData,
    JudgeErrorFitResult,
    build_model,
    fit_nuts,
    fit_svi,
    posterior_winrate_interval,
)

__all__ = [
    "N_OUTCOMES",
    "OUTCOME_CLASSES",
    "JudgeErrorData",
    "JudgeErrorFitResult",
    "build_model",
    "fit_nuts",
    "fit_svi",
    "posterior_winrate_interval",
]
