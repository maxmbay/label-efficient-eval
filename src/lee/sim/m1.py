"""Milestone 1 simulation: ToxicChat prevalence under label budgets.

Fixed seeds (locked for runs):
  SEED_ROOT = 20261026
  For labeled subset draw r at budget n:
    seed = SEED_ROOT + 1_000_003 * n + r
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from lee.data.toxicchat import load_toxicchat
from lee.estimators import (
    classical_clt,
    classical_wilson,
    judge_only_mean,
    ppi_ci,
    ppi_pp_ci,
)
from lee.judges import ACTIVE_M1_JUDGES
from lee.judges.cache import align_scores, judge_cache_dir, load_scores
from lee.sim.headline import compute_headline
from lee.sim.metrics import SimMetrics, summarize_runs

SEED_ROOT = 20261026
DEFAULT_N_GRID = (50, 100, 200, 400, 800, 1600)
DEFAULT_ALPHAS = (0.10, 0.05)  # nominal 90% and 95%
DEFAULT_ESTIMANDS = ("toxicity", "jailbreaking")


def subset_seed(n: int, r: int) -> int:
    return int(SEED_ROOT + 1_000_003 * n + r)


@dataclass(frozen=True)
class M1Config:
    n_grid: tuple[int, ...] = DEFAULT_N_GRID
    R: int = 1000
    alphas: tuple[float, ...] = DEFAULT_ALPHAS
    estimands: tuple[str, ...] = DEFAULT_ESTIMANDS
    seed_root: int = SEED_ROOT
    judge_threshold: float = 0.5  # unused for soft judge-only mean


def _truth(y: np.ndarray) -> float:
    return float(np.asarray(y, dtype=float).mean())


def _run_one_draw(
    *,
    y: np.ndarray,
    f: np.ndarray,
    labeled_idx: np.ndarray,
    unlabeled_idx: np.ndarray,
    alpha: float,
    f_all: np.ndarray,
) -> dict[str, tuple[float, float, float, float | None]]:
    """Return method -> (estimate, lower, upper, lam)."""
    y_l = y[labeled_idx]
    f_l = f[labeled_idx]
    f_u = f[unlabeled_idx]
    out: dict[str, tuple[float, float, float, float | None]] = {}

    clt = classical_clt(y_l, alpha=alpha)
    out["classical_clt"] = (clt.estimate, clt.lower, clt.upper, 0.0)

    wil = classical_wilson(y_l, alpha=alpha)
    out["classical_wilson"] = (wil.estimate, wil.lower, wil.upper, 0.0)

    jo = judge_only_mean(f_all, alpha=alpha)
    out["judge_only"] = (jo.estimate, jo.lower, jo.upper, None)

    ppi = ppi_ci(y_l, f_l, f_u, alpha=alpha)
    out["ppi"] = (ppi.estimate, ppi.lower, ppi.upper, ppi.lam)

    pp = ppi_pp_ci(y_l, f_l, f_u, alpha=alpha)
    out["ppi_pp"] = (pp.estimate, pp.lower, pp.upper, pp.lam)
    return out


def run_m1_grid(
    *,
    population: pd.DataFrame,
    judge_scores: dict[str, np.ndarray],
    config: M1Config | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[SimMetrics]:
    """Run the M1 subsampling grid; return one SimMetrics row per cell."""
    cfg = config or M1Config()
    log = progress or (lambda _msg: None)
    results: list[SimMetrics] = []
    n_pop = len(population)

    for estimand in cfg.estimands:
        y = population[estimand].to_numpy(dtype=float)
        truth = _truth(y)
        for judge_name, f in judge_scores.items():
            if not _judge_matches_estimand(judge_name, estimand):
                continue
            f = np.asarray(f, dtype=float)
            if f.shape[0] != n_pop:
                raise ValueError(
                    f"Judge {judge_name} length {f.shape[0]} != N={n_pop}"
                )
            for n in cfg.n_grid:
                if n >= n_pop:
                    log(f"skip n={n} >= N={n_pop}")
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
                            labeled_idx=labeled,
                            unlabeled_idx=unlabeled,
                            alpha=alpha,
                            f_all=f,
                        )
                        for method, (est, lo, hi, lam) in draws.items():
                            b = buckets.setdefault(
                                method,
                                {
                                    "est": [],
                                    "lo": [],
                                    "hi": [],
                                    "lam": [],
                                },
                            )
                            b["est"].append(est)
                            b["lo"].append(lo)
                            b["hi"].append(hi)
                            b["lam"].append(
                                np.nan if lam is None else float(lam)
                            )
                    for method, b in buckets.items():
                        lams = np.asarray(b["lam"], dtype=float)
                        mean_lam = (
                            None
                            if np.isnan(lams).all()
                            else float(np.nanmean(lams))
                        )
                        metrics = summarize_runs(
                            estimates=np.asarray(b["est"]),
                            lowers=np.asarray(b["lo"]),
                            uppers=np.asarray(b["hi"]),
                            truth=truth,
                            alpha=alpha,
                            estimator=method,
                            judge=judge_name,
                            estimand=estimand,
                            n=n,
                            lams=None if mean_lam is None else lams,
                        )
                        results.append(metrics)
                log(
                    f"done estimand={estimand} judge={judge_name} n={n} R={cfg.R}"
                )
    return results


def _judge_matches_estimand(judge_name: str, estimand: str) -> bool:
    """Estimand-specific judges (open_it_*) only run on their target label."""
    if judge_name.startswith("open_it_toxicity"):
        return estimand == "toxicity"
    if judge_name.startswith("open_it_jailbreaking"):
        return estimand == "jailbreaking"
    return True


def score_active_judges(population: pd.DataFrame) -> dict[str, np.ndarray]:
    """Score cheap in-process judges (openai_mod, weak keyword)."""
    scores: dict[str, np.ndarray] = {}
    for cls in ACTIVE_M1_JUDGES:
        judge = cls()
        try:
            scores[judge.name] = judge.score(population)
        except NotImplementedError as exc:
            raise RuntimeError(str(exc)) from exc
    return scores


def load_cached_judges(population: pd.DataFrame) -> dict[str, np.ndarray]:
    """Load any parquet caches under data/cache/judges/."""
    scores: dict[str, np.ndarray] = {}
    root = judge_cache_dir()
    for path in sorted(root.glob("*.parquet")):
        name = path.stem
        cached = load_scores(name)
        if cached is None:
            continue
        scores[name] = align_scores(population, cached)
    return scores


def collect_judge_scores(population: pd.DataFrame) -> dict[str, np.ndarray]:
    """Merge in-process judges with on-disk cached ladder scores."""
    scores = score_active_judges(population)
    cached = load_cached_judges(population)
    overlap = set(scores) & set(cached)
    if overlap:
        # Prefer cache for names that exist on disk (reproducible ladder runs).
        for name in overlap:
            scores[name] = cached[name]
    for name, arr in cached.items():
        scores[name] = arr
    return scores


def metrics_to_frame(metrics: Iterable[SimMetrics]) -> pd.DataFrame:
    return pd.DataFrame([m.to_dict() for m in metrics])


def default_output_dir() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "outputs" / "m1"


def run_and_save(
    *,
    R: int = 1000,
    n_grid: tuple[int, ...] | None = None,
    out_dir: Path | None = None,
    smoke: bool = False,
) -> Path:
    """Load data, run grid, write CSV + JSON sidecar. Returns output directory."""
    out = out_dir or default_output_dir()
    out.mkdir(parents=True, exist_ok=True)

    _all_rows, population, report = load_toxicchat(strict=True)
    (out / "load_report.json").write_text(report.to_json() + "\n")

    judge_scores = collect_judge_scores(population)
    # Cache judge vectors (floats only; no raw text).
    cache = out / "judge_scores"
    cache.mkdir(exist_ok=True)
    for name, arr in judge_scores.items():
        np.save(cache / f"{name}.npy", arr)

    cfg = M1Config(
        R=R,
        n_grid=tuple(n_grid) if n_grid is not None else DEFAULT_N_GRID,
    )
    tag = "smoke" if smoke or R < 1000 else "full"

    def _progress(msg: str) -> None:
        print(msg, flush=True)

    metrics = run_m1_grid(
        population=population,
        judge_scores=judge_scores,
        config=cfg,
        progress=_progress,
    )
    frame = metrics_to_frame(metrics)
    csv_path = out / f"metrics_{tag}.csv"
    frame.to_csv(csv_path, index=False)
    (out / f"metrics_{tag}.json").write_text(
        json.dumps([asdict(m) for m in metrics], indent=2) + "\n"
    )

    headline = compute_headline(frame, estimand="toxicity")
    meta = {
        "tag": tag,
        "R": cfg.R,
        "n_grid": list(cfg.n_grid),
        "seed_root": cfg.seed_root,
        "N": len(population),
        "truth": {
            e: float(population[e].astype(float).mean()) for e in cfg.estimands
        },
        "judges": list(judge_scores),
        "headline_n": 200,
        "headline": headline.to_dict(),
    }
    (out / f"run_meta_{tag}.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out / f"headline_{tag}.json").write_text(
        json.dumps(headline.to_dict(), indent=2) + "\n"
    )
    print(headline.message, flush=True)
    print(f"Wrote {csv_path}", flush=True)
    return out
