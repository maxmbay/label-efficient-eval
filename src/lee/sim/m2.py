"""Milestone 2 simulation: MT-Bench pairwise win rates under label budgets.

Fixed seeds (locked for runs):
  SEED_ROOT = 20261026
  For labeled subset draw r at budget n:
    seed = SEED_ROOT + 2_000_003 * n + r
  (M2 offset; distinct from M1.)

Primary pair after load: gpt-3.5-turbo vs gpt-4 (see PRIMARY_PAIR).
Decisive N=105, so the plan's n=200 headline budget is unavailable; the
amended headline budget is n=50 (largest grid point with unlabeled remainder).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from lee.data.mt_bench import (
    PRIMARY_PAIR,
    build_pair_population,
    load_mt_bench,
)
from lee.estimators.classical import classical_clt, classical_clustered
from lee.estimators.judge_only import judge_only_mean
from lee.estimators.ppi import ppi_ci, ppi_pp_ci
from lee.sim.metrics import SimMetrics, summarize_runs

SEED_ROOT = 20261026
# Full plan grid; entries with n >= N are skipped at runtime.
DEFAULT_N_GRID = (50, 100, 200, 400, 800, 1600)
# Smoke / pair-feasible grid for N≈105.
SMOKE_N_GRID = (30, 50, 80)
DEFAULT_ALPHAS = (0.10, 0.05)
HEADLINE_N = 50  # amended after load; plan n=200 exceeds pair N
JUDGE_NAME = "gpt4_pair"


def subset_seed(n: int, r: int) -> int:
    return int(SEED_ROOT + 2_000_003 * n + r)


@dataclass(frozen=True)
class M2Config:
    n_grid: tuple[int, ...] = DEFAULT_N_GRID
    R: int = 1000
    alphas: tuple[float, ...] = DEFAULT_ALPHAS
    seed_root: int = SEED_ROOT
    model_lo: str = PRIMARY_PAIR[0]
    model_hi: str = PRIMARY_PAIR[1]
    headline_n: int = HEADLINE_N


def _truth(y: np.ndarray) -> float:
    return float(np.asarray(y, dtype=float).mean())


def _run_one_draw(
    *,
    y: np.ndarray,
    f: np.ndarray,
    question_ids: np.ndarray,
    labeled_idx: np.ndarray,
    unlabeled_idx: np.ndarray,
    alpha: float,
    f_all: np.ndarray,
) -> dict[str, tuple[float, float, float, float | None]]:
    y_l = y[labeled_idx]
    f_l = f[labeled_idx]
    f_u = f[unlabeled_idx]
    q_l = question_ids[labeled_idx]
    out: dict[str, tuple[float, float, float, float | None]] = {}

    clt = classical_clt(y_l, alpha=alpha)
    out["classical_clt"] = (clt.estimate, clt.lower, clt.upper, 0.0)

    clus = classical_clustered(y_l, q_l, alpha=alpha)
    out["classical_clustered"] = (clus.estimate, clus.lower, clus.upper, 0.0)

    jo = judge_only_mean(f_all, alpha=alpha)
    out["judge_only"] = (jo.estimate, jo.lower, jo.upper, None)

    ppi = ppi_ci(y_l, f_l, f_u, alpha=alpha)
    out["ppi"] = (ppi.estimate, ppi.lower, ppi.upper, ppi.lam)

    pp = ppi_pp_ci(y_l, f_l, f_u, alpha=alpha)
    out["ppi_pp"] = (pp.estimate, pp.lower, pp.upper, pp.lam)
    return out


def run_m2_grid(
    *,
    population: pd.DataFrame,
    config: M2Config | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[SimMetrics]:
    """Run the M2 subsampling grid on one pairwise population."""
    cfg = config or M2Config()
    log = progress or (lambda _msg: None)
    y = population["y"].to_numpy(dtype=float)
    f = population["f"].to_numpy(dtype=float)
    q = population["question_id"].to_numpy()
    n_pop = len(population)
    truth = _truth(y)
    estimand = f"winrate:{cfg.model_lo}_vs_{cfg.model_hi}"
    results: list[SimMetrics] = []

    for n in cfg.n_grid:
        if n >= n_pop:
            log(f"skip n={n} >= N={n_pop}")
            continue
        if n_pop - n < 1:
            log(f"skip n={n}: no unlabeled remainder")
            continue
        for alpha in cfg.alphas:
            buckets: dict[str, dict[str, list[float]]] = {}
            for r in range(cfg.R):
                rng = np.random.default_rng(subset_seed(n, r))
                labeled = rng.choice(n_pop, size=n, replace=False)
                mask = np.ones(n_pop, dtype=bool)
                mask[labeled] = False
                unlabeled = np.flatnonzero(mask)
                draws = _run_one_draw(
                    y=y,
                    f=f,
                    question_ids=q,
                    labeled_idx=labeled,
                    unlabeled_idx=unlabeled,
                    alpha=alpha,
                    f_all=f,
                )
                for method, (est, lo, hi, lam) in draws.items():
                    b = buckets.setdefault(
                        method, {"est": [], "lo": [], "hi": [], "lam": []}
                    )
                    b["est"].append(est)
                    b["lo"].append(lo)
                    b["hi"].append(hi)
                    b["lam"].append(np.nan if lam is None else float(lam))
            for method, b in buckets.items():
                lams = np.asarray(b["lam"], dtype=float)
                mean_lam = None if np.isnan(lams).all() else float(np.nanmean(lams))
                results.append(
                    summarize_runs(
                        estimates=np.asarray(b["est"]),
                        lowers=np.asarray(b["lo"]),
                        uppers=np.asarray(b["hi"]),
                        truth=truth,
                        alpha=alpha,
                        estimator=method,
                        judge=JUDGE_NAME,
                        estimand=estimand,
                        n=n,
                        lams=None if mean_lam is None else lams,
                    )
                )
            log(f"done estimand={estimand} n={n} alpha={alpha} R={cfg.R}")
    return results


def metrics_to_frame(metrics: Iterable[SimMetrics]) -> pd.DataFrame:
    return pd.DataFrame([m.to_dict() for m in metrics])


def default_output_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "outputs" / "m2"


def run_and_save(
    *,
    R: int = 1000,
    n_grid: tuple[int, ...] | None = None,
    out_dir: Path | None = None,
    smoke: bool = False,
) -> Path:
    """Load MT-Bench pair population, run grid, write CSV + JSON."""
    out = out_dir or default_output_dir()
    out.mkdir(parents=True, exist_ok=True)

    human, gpt4, _cats, report = load_mt_bench(strict=True)
    (out / "load_report.json").write_text(report.to_json() + "\n")

    model_lo, model_hi = PRIMARY_PAIR
    population = build_pair_population(
        human, gpt4, model_lo=model_lo, model_hi=model_hi
    )
    population.to_parquet(out / "pair_population.parquet", index=False)

    grid = (
        tuple(n_grid)
        if n_grid is not None
        else (SMOKE_N_GRID if smoke else DEFAULT_N_GRID)
    )
    cfg = M2Config(R=R, n_grid=grid, model_lo=model_lo, model_hi=model_hi)
    tag = "smoke" if smoke or R < 1000 else "full"

    def _progress(msg: str) -> None:
        print(msg, flush=True)

    metrics = run_m2_grid(population=population, config=cfg, progress=_progress)
    frame = metrics_to_frame(metrics)
    csv_path = out / f"metrics_{tag}.csv"
    frame.to_csv(csv_path, index=False)
    (out / f"metrics_{tag}.json").write_text(
        json.dumps([asdict(m) for m in metrics], indent=2) + "\n"
    )

    truth = float(population["y"].mean())
    meta = {
        "tag": tag,
        "R": cfg.R,
        "n_grid": list(cfg.n_grid),
        "n_grid_run": sorted({m.n for m in metrics}),
        "seed_root": cfg.seed_root,
        "subset_seed_formula": "SEED_ROOT + 2_000_003 * n + r",
        "N": len(population),
        "pair": [model_lo, model_hi],
        "truth_winrate_model_lo": truth,
        "judge": JUDGE_NAME,
        "headline_n_plan": 200,
        "headline_n_amended": cfg.headline_n,
        "headline_note": (
            "Plan n=200 exceeds decisive N for the primary pair; "
            f"amended headline budget is n={cfg.headline_n}."
        ),
        "inter_human_agree_nontie": report.inter_human_agree_nontie,
    }
    (out / f"run_meta_{tag}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)
    print(f"Wrote {csv_path}", flush=True)
    return out
