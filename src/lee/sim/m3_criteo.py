"""Milestone 3 Criteo CUPAC bridge: ATE on visit, A/A, PPI identity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lee.data.criteo import (
    DEFAULT_SUBSAMPLE_N,
    FEATURE_COLS,
    SEED_ROOT,
    feature_matrix,
    load_criteo,
)
from lee.estimators.cupac import (
    cupac_from_features,
    diff_in_means_ate,
    ppi_form_mean_from_cupac,
)

R_AA_FULL = 200
R_AA_SMOKE = 20


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_out_dir() -> Path:
    return _repo_root() / "outputs" / "m3"


def run_aa(
    df: pd.DataFrame,
    *,
    R: int,
    seed: int = SEED_ROOT + 17,
    max_control: int = 100_000,
) -> dict:
    """A/A on control only: fake arms, FPR at alpha=0.05 for DIM and CUPAC.

    Fits one cross-fit outcome model on control features (outcome model does
    not use treatment), then for each replicate only re-randomizes fake arms
    and recomputes DIM / CUPAC on the fixed predictions. This matches the
    CUPAC form under A/A without refitting the ML model R times.
    """
    from lee.estimators.cupac import crossfit_outcome_scores, cupac_ate

    ctrl = df.loc[df["treatment"] == 0].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    if len(ctrl) > max_control:
        take = rng.choice(len(ctrl), size=max_control, replace=False)
        ctrl = ctrl.iloc[take].reset_index(drop=True)
    y = ctrl["visit"].to_numpy(dtype=float)
    x = feature_matrix(ctrl)
    f = crossfit_outcome_scores(x, y, n_folds=3, seed=seed)

    dim_reject = 0
    cupac_reject = 0
    for r in range(R):
        rr = np.random.default_rng(seed + 1 + r)
        fake_t = rr.integers(0, 2, size=len(ctrl)).astype(float)
        dim = diff_in_means_ate(y, fake_t, alpha=0.05)
        cupac = cupac_ate(y, fake_t, f, alpha=0.05)
        dim_reject += int(dim.rejects_zero())
        cupac_reject += int(cupac.rejects_zero())

    return {
        "R": R,
        "n_control": int(len(ctrl)),
        "max_control": max_control,
        "dim_fpr": dim_reject / R,
        "cupac_fpr": cupac_reject / R,
        "alpha": 0.05,
        "seed": seed,
        "note": (
            "Outcome model cross-fit once on control; fake arms re-drawn each "
            "replicate. CUPAC adjustment uses fixed oof predictions."
        ),
    }


def run_criteo(
    *,
    smoke: bool = False,
    subsample_n: int | None = DEFAULT_SUBSAMPLE_N,
    full: bool = False,
    r_aa: int | None = None,
) -> dict:
    df, load_report = load_criteo(
        subsample_n=None if full else subsample_n,
        full=full,
        seed=SEED_ROOT,
    )
    y = df["visit"].to_numpy(dtype=float)
    t = df["treatment"].to_numpy(dtype=float)
    x = feature_matrix(df)

    dim = diff_in_means_ate(y, t, alpha=0.05)
    cupac, f_hat, lam = cupac_from_features(
        x, y, t, n_folds=5, seed=SEED_ROOT, alpha=0.05
    )

    # Numerical CUPAC ≡ PPI identity on a mean split of visit vs f_hat.
    cupac_like, ppi_like, lam_id = ppi_form_mean_from_cupac(y, f_hat, lam=None)
    abs_err = abs(cupac_like - ppi_like)
    rel_err = abs_err / max(abs(ppi_like), 1e-15)

    var_ratio = dim.var / cupac.var if cupac.var > 0 else float("nan")
    aa_R = r_aa if r_aa is not None else (R_AA_SMOKE if smoke else R_AA_FULL)
    aa = run_aa(df, R=aa_R)

    result = {
        "smoke": smoke,
        "load_report": json.loads(load_report.to_json()),
        "feature_cols": list(FEATURE_COLS),
        "dim": {
            "estimate": dim.estimate,
            "se": dim.se,
            "lower": dim.lower,
            "upper": dim.upper,
            "var": dim.var,
            "method": dim.method,
        },
        "cupac": {
            "estimate": cupac.estimate,
            "se": cupac.se,
            "lower": cupac.lower,
            "upper": cupac.upper,
            "var": cupac.var,
            "method": cupac.method,
            "lam": lam,
        },
        "variance_ratio_dim_over_cupac": var_ratio,
        "equivalent_n_factor": var_ratio,
        "ppi_identity": {
            "cupac_like": cupac_like,
            "ppi_like": ppi_like,
            "lam": lam_id,
            "abs_err": abs_err,
            "rel_err": rel_err,
            "pass": bool(abs_err <= 1e-12 or rel_err <= 1e-10),
        },
        "aa": aa,
        "headline": {
            "variance_reduction_factor": var_ratio,
            "equivalent_n_factor": var_ratio,
            "dim_ate": dim.estimate,
            "cupac_ate": cupac.estimate,
            "aa_dim_fpr": aa["dim_fpr"],
            "aa_cupac_fpr": aa["cupac_fpr"],
            "subsampled": load_report.subsampled,
            "n_rows_used": load_report.n_rows_used,
            "n_rows_file": load_report.n_rows_file,
        },
        "databricks_status": (
            "not_configured_on_this_machine; local pandas subsample path used"
            if load_report.subsampled
            else "full pandas load; Spark notebook still provided for Databricks"
        ),
    }
    return result


def run_and_save(
    *,
    smoke: bool = False,
    subsample_n: int | None = DEFAULT_SUBSAMPLE_N,
    full: bool = False,
    out_dir: Path | None = None,
) -> Path:
    out = out_dir or default_out_dir()
    out.mkdir(parents=True, exist_ok=True)
    # Smoke uses a smaller subsample for wall-clock.
    if smoke and not full:
        subsample_n = min(subsample_n or DEFAULT_SUBSAMPLE_N, 50_000)
    result = run_criteo(smoke=smoke, subsample_n=subsample_n, full=full)
    tag = "smoke" if smoke else "full"
    path = out / f"criteo_{tag}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    # Compact CSV of main estimates.
    rows = [
        {
            "estimator": "dim",
            "ate": result["dim"]["estimate"],
            "se": result["dim"]["se"],
            "var": result["dim"]["var"],
        },
        {
            "estimator": "cupac",
            "ate": result["cupac"]["estimate"],
            "se": result["cupac"]["se"],
            "var": result["cupac"]["var"],
        },
    ]
    pd.DataFrame(rows).to_csv(out / f"criteo_{tag}.csv", index=False)
    return path
