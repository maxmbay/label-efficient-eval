"""Bradley-Terry strengths for MT-Bench model comparison.

Point estimates via MLE on non-tie comparisons. Uncertainty via a
question-level cluster bootstrap (resampling question_id blocks).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import optimize


@dataclass(frozen=True)
class BradleyTerryResult:
    """Strengths are identified up to an additive constant (mean-zero)."""

    strengths: dict[str, float]
    se: dict[str, float]
    pairwise_prob: dict[tuple[str, str], float]
    n_comparisons: int
    n_questions: int
    method: str
    notes: tuple[str, ...]

    def to_dict(self) -> dict:
        d = asdict(self)
        # JSON-friendly pair keys
        d["pairwise_prob"] = {
            f"{a}__vs__{b}": p for (a, b), p in self.pairwise_prob.items()
        }
        return d


def _design_from_frame(
    frame: pd.DataFrame,
    *,
    model_a_col: str = "model_lo",
    model_b_col: str = "model_hi",
    winner_col: str = "majority_winner",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (winner_is_a, idx_a, idx_b, model_names).

    ``winner_col`` values are ``model_a`` / ``model_b`` in the same orientation
    as model_lo / model_hi (canonical), or raw model names.
    """
    models = sorted(
        set(frame[model_a_col]).union(frame[model_b_col])
    )
    index = {m: i for i, m in enumerate(models)}
    a = frame[model_a_col].map(index).to_numpy(dtype=int)
    b = frame[model_b_col].map(index).to_numpy(dtype=int)
    w = frame[winner_col].astype(str)
    # Map to whether model_a (lo) won
    if set(w.unique()) <= {"model_a", "model_b"}:
        y = (w == "model_a").to_numpy(dtype=float)
    else:
        y = (w.to_numpy() == frame[model_a_col].to_numpy()).astype(float)
    return y, a, b, models


def fit_bradley_terry_mle(
    y: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    *,
    n_models: int,
) -> np.ndarray:
    """MLE strengths with mean-zero constraint. Returns length-n_models array."""
    y = np.asarray(y, dtype=float).ravel()
    idx_a = np.asarray(idx_a, dtype=int).ravel()
    idx_b = np.asarray(idx_b, dtype=int).ravel()

    def nll(theta_free: np.ndarray) -> float:
        # theta_free has length n_models-1; last = -sum for mean zero
        theta = np.zeros(n_models)
        theta[:-1] = theta_free
        theta[-1] = -float(theta_free.sum())
        diff = theta[idx_a] - theta[idx_b]
        # P(A beats B) = sigmoid(theta_a - theta_b)
        # nll = -sum [ y*diff - softplus(diff) ]
        return float(np.sum(np.logaddexp(0.0, diff) - y * diff))

    def grad(theta_free: np.ndarray) -> np.ndarray:
        theta = np.zeros(n_models)
        theta[:-1] = theta_free
        theta[-1] = -float(theta_free.sum())
        diff = theta[idx_a] - theta[idx_b]
        p = 1.0 / (1.0 + np.exp(-diff))
        resid = p - y  # d/d(diff) of nll
        g_full = np.zeros(n_models)
        np.add.at(g_full, idx_a, resid)
        np.add.at(g_full, idx_b, -resid)
        # chain rule for mean-zero: d theta[-1] / d theta_free[k] = -1
        g_free = g_full[:-1] - g_full[-1]
        return g_free

    x0 = np.zeros(n_models - 1)
    res = optimize.minimize(nll, x0, jac=grad, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(f"Bradley-Terry MLE failed: {res.message}")
    theta = np.zeros(n_models)
    theta[:-1] = res.x
    theta[-1] = -float(res.x.sum())
    return theta


def _pairwise_probs(theta: np.ndarray, models: list[str]) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if i >= j:
                continue
            out[(a, b)] = float(1.0 / (1.0 + np.exp(-(theta[i] - theta[j]))))
    return out


def fit_bradley_terry(
    frame: pd.DataFrame,
    *,
    model_a_col: str = "model_lo",
    model_b_col: str = "model_hi",
    winner_col: str = "majority_winner",
    question_col: str = "question_id",
    n_bootstrap: int = 500,
    seed: int = 20261026,
) -> BradleyTerryResult:
    """Fit BT on non-tie rows; cluster-bootstrap SEs by question_id."""
    use = frame.copy()
    w = use[winner_col].astype(str)
    if set(w.unique()) <= {"model_a", "model_b", "tie", "majority_tie"}:
        use = use[w.isin(["model_a", "model_b"])].copy()
    else:
        use = use[w.isin(set(use[model_a_col]).union(use[model_b_col]))].copy()
        use = use[use[winner_col] != "tie"].copy()

    if use.empty:
        raise ValueError("No non-tie comparisons for Bradley-Terry")

    y, idx_a, idx_b, models = _design_from_frame(
        use,
        model_a_col=model_a_col,
        model_b_col=model_b_col,
        winner_col=winner_col,
    )
    theta = fit_bradley_terry_mle(y, idx_a, idx_b, n_models=len(models))

    # Cluster bootstrap
    questions = use[question_col].to_numpy()
    uniq_q = np.unique(questions)
    rng = np.random.default_rng(seed)
    boot = np.zeros((n_bootstrap, len(models)))
    for b in range(n_bootstrap):
        sampled_q = rng.choice(uniq_q, size=uniq_q.size, replace=True)
        # gather rows whose question is in sampled_q (with multiplicity)
        parts = []
        for q in sampled_q:
            parts.append(use.loc[questions == q])
        boot_frame = pd.concat(parts, ignore_index=True)
        # drop models missing from a bootstrap replicate by skipping bad fits
        try:
            y_b, a_b, b_b, models_b = _design_from_frame(
                boot_frame,
                model_a_col=model_a_col,
                model_b_col=model_b_col,
                winner_col=winner_col,
            )
            if models_b != models:
                # align into full model list
                theta_b_full = np.zeros(len(models))
                theta_sub = fit_bradley_terry_mle(
                    y_b, a_b, b_b, n_models=len(models_b)
                )
                for m, val in zip(models_b, theta_sub):
                    theta_b_full[models.index(m)] = val
                # re-center
                theta_b_full -= theta_b_full.mean()
                boot[b] = theta_b_full
            else:
                boot[b] = fit_bradley_terry_mle(
                    y_b, a_b, b_b, n_models=len(models)
                )
        except RuntimeError:
            boot[b] = np.nan

    se = np.nanstd(boot, axis=0, ddof=1)
    strengths = {m: float(t) for m, t in zip(models, theta)}
    se_map = {m: float(s) for m, s in zip(models, se)}
    notes = (
        "Strengths mean-zero; P(i>j)=sigmoid(theta_i-theta_j).",
        f"SEs from question_id cluster bootstrap (B={n_bootstrap}).",
        "Fit on non-tie comparisons only.",
    )
    return BradleyTerryResult(
        strengths=strengths,
        se=se_map,
        pairwise_prob=_pairwise_probs(theta, models),
        n_comparisons=int(len(use)),
        n_questions=int(uniq_q.size),
        method="bradley_terry_mle_cluster_bootstrap",
        notes=notes,
    )
