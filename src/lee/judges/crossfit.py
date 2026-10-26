"""Cross-fit calibration so predictors stay independent of the labeled fold.

Fit Platt scaling (logistic regression on score -> label) on train, apply to
test; reverse; concatenate. Required whenever labels are used to calibrate or
threshold a judge (Zrnic and Candès, cross-prediction-powered inference).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def crossfit_platt(
    population: pd.DataFrame,
    raw_scores: np.ndarray,
    *,
    label_col: str,
    split_col: str = "split",
) -> np.ndarray:
    """Return calibrated probabilities with train/test cross-fitting."""
    raw = np.asarray(raw_scores, dtype=float).ravel()
    if len(population) != raw.size:
        raise ValueError("population/raw_scores length mismatch")
    if split_col not in population.columns:
        raise KeyError(f"missing split column {split_col!r}")

    out = np.full(raw.size, np.nan, dtype=float)
    splits = population[split_col].astype(str).to_numpy()
    y = population[label_col].to_numpy(dtype=float)

    for fit_split, apply_split in (("train", "test"), ("test", "train")):
        fit_mask = splits == fit_split
        apply_mask = splits == apply_split
        if fit_mask.sum() < 2 or apply_mask.sum() < 1:
            raise ValueError(f"Insufficient rows for cross-fit ({fit_split}->{apply_split})")
        x_fit = raw[fit_mask].reshape(-1, 1)
        y_fit = y[fit_mask]
        # Degenerate labels: fall back to raw scores on apply fold.
        if len(np.unique(y_fit)) < 2:
            out[apply_mask] = raw[apply_mask]
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(x_fit, y_fit)
        out[apply_mask] = clf.predict_proba(raw[apply_mask].reshape(-1, 1))[:, 1]

    if np.isnan(out).any():
        raise RuntimeError("cross-fit left NaNs; check split labels")
    return out


def maybe_crossfit(
    population: pd.DataFrame,
    raw_scores: np.ndarray,
    *,
    label_col: str,
    calibrate: bool,
) -> tuple[np.ndarray, bool]:
    """If ``calibrate``, return cross-fit Platt scores; else raw."""
    if not calibrate:
        return np.asarray(raw_scores, dtype=float), False
    return crossfit_platt(population, raw_scores, label_col=label_col), True
