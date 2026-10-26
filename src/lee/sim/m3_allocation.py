"""Milestone 3 allocation simulation: active vs uniform AIPW on ToxicChat."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lee.data.toxicchat import load_toxicchat
from lee.estimators.active import (
    aipw_mean_ci,
    make_propensity,
    sample_labels,
    uniform_propensity,
)
from lee.judges.openai_moderation import OpenAIModerationMaxJudge

SEED_ROOT = 20261026
DEFAULT_N_GRID = (50, 100, 200, 400, 800)
ALPHAS = (0.05, 0.10)


def allocation_seed(n_budget: int, r: int) -> int:
    return SEED_ROOT + 3_000_007 * int(n_budget) + int(r)


@dataclass(frozen=True)
class AllocationMetrics:
    method: str
    n_budget: int
    alpha: float
    R: int
    coverage: float
    coverage_se: float
    mean_width: float
    bias: float
    rmse: float
    mean_n_labeled: float


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_out_dir() -> Path:
    return _repo_root() / "outputs" / "m3"


def run_allocation(
    *,
    R: int = 500,
    n_grid: tuple[int, ...] = DEFAULT_N_GRID,
    smoke: bool = False,
) -> tuple[list[AllocationMetrics], dict]:
    _, human, load_report = load_toxicchat(strict=True)
    y = human["toxicity"].astype(float).to_numpy()
    f = OpenAIModerationMaxJudge().score(human)
    truth = float(y.mean())
    n_pop = y.size

    rows: list[AllocationMetrics] = []
    for n_budget in n_grid:
        for method in ("uniform", "active"):
            for alpha in ALPHAS:
                estimates = np.empty(R)
                lowers = np.empty(R)
                uppers = np.empty(R)
                n_lab = np.empty(R)
                for r in range(R):
                    rng = np.random.default_rng(allocation_seed(n_budget, r))
                    if method == "uniform":
                        pi = uniform_propensity(n_pop, n_budget=n_budget)
                    else:
                        pi = make_propensity(f, n_budget=n_budget)
                    xi = sample_labels(pi, rng=rng).astype(float)
                    # Labels only observed when xi=1; AIPW uses Y everywhere in
                    # the formula, which is valid in simulation where Y is known.
                    est = aipw_mean_ci(
                        y, f, xi, pi, alpha=alpha, method=f"aipw_{method}"
                    )
                    estimates[r] = est.estimate
                    lowers[r] = est.lower
                    uppers[r] = est.upper
                    n_lab[r] = float(xi.sum())
                covered = (lowers <= truth) & (truth <= uppers)
                cov = float(covered.mean())
                cov_se = float(np.sqrt(cov * (1.0 - cov) / R))
                rows.append(
                    AllocationMetrics(
                        method=method,
                        n_budget=int(n_budget),
                        alpha=float(alpha),
                        R=R,
                        coverage=cov,
                        coverage_se=cov_se,
                        mean_width=float((uppers - lowers).mean()),
                        bias=float(estimates.mean() - truth),
                        rmse=float(np.sqrt(np.mean((estimates - truth) ** 2))),
                        mean_n_labeled=float(n_lab.mean()),
                    )
                )

    headline = _allocation_headline(rows, truth=truth)
    meta = {
        "seed_root": SEED_ROOT,
        "R": R,
        "n_grid": list(n_grid),
        "smoke": smoke,
        "truth_toxicity": truth,
        "n_population": n_pop,
        "judge": OpenAIModerationMaxJudge.name,
        "load_report": json.loads(load_report.to_json()),
        "headline": headline,
    }
    return rows, meta


def _allocation_headline(rows: list[AllocationMetrics], *, truth: float) -> dict:
    """Locked headline at n_b=200, alpha=0.05."""
    del truth  # reserved for future diagnostics
    target_n = 200
    alpha = 0.05
    by = {(m.method, m.n_budget, m.alpha): m for m in rows}
    u = by.get(("uniform", target_n, alpha))
    a = by.get(("active", target_n, alpha))
    if u is None or a is None:
        return {"status": "missing_rows", "n_budget": target_n, "alpha": alpha}
    # Coverage within MC SE of 95%.
    ok_u = abs(u.coverage - 0.95) <= max(2.0 * u.coverage_se, 0.02)
    ok_a = abs(a.coverage - 0.95) <= max(2.0 * a.coverage_se, 0.02)
    if not (ok_u and ok_a):
        return {
            "status": "coverage_fail",
            "n_budget": target_n,
            "alpha": alpha,
            "uniform_coverage": u.coverage,
            "uniform_coverage_se": u.coverage_se,
            "active_coverage": a.coverage,
            "active_coverage_se": a.coverage_se,
            "uniform_mean_width": u.mean_width,
            "active_mean_width": a.mean_width,
        }
    rel = (u.mean_width - a.mean_width) / u.mean_width if u.mean_width else float("nan")
    return {
        "status": "ok",
        "n_budget": target_n,
        "alpha": alpha,
        "uniform_mean_width": u.mean_width,
        "active_mean_width": a.mean_width,
        "relative_width_reduction": rel,
        "uniform_coverage": u.coverage,
        "active_coverage": a.coverage,
    }


def run_and_save(
    *,
    R: int = 500,
    n_grid: tuple[int, ...] | None = None,
    smoke: bool = False,
    out_dir: Path | None = None,
) -> Path:
    out = out_dir or default_out_dir()
    out.mkdir(parents=True, exist_ok=True)
    grid = n_grid or DEFAULT_N_GRID
    rows, meta = run_allocation(R=R, n_grid=grid, smoke=smoke)
    tag = "smoke" if smoke or R < 500 else "full"
    df = pd.DataFrame([asdict(r) for r in rows])
    csv_path = out / f"allocation_{tag}.csv"
    json_path = out / f"allocation_{tag}.json"
    meta_path = out / f"allocation_meta_{tag}.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2) + "\n")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return csv_path
