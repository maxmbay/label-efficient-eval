"""Milestone 3 stopping simulation: betting CS on PPI rectifier stream."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lee.data.toxicchat import load_toxicchat
from lee.estimators.betting_cs import (
    HedgedBettingCS,
    affine_from_unit,
    affine_to_unit,
    first_clear_above,
    first_excludes,
    ppi_rectifier_bounds,
)
from lee.judges.openai_moderation import OpenAIModerationMaxJudge

SEED_ROOT = 20261026
TAU = 0.10
T_MAX = 800
ALPHA = 0.05
LAM = 1.0


def stopping_seed(r: int) -> int:
    return SEED_ROOT + 4_000_009 * int(r)


@dataclass(frozen=True)
class StoppingSummary:
    rule: str
    R: int
    alpha: float
    tau: float
    t_max: int
    median_T: float
    mean_T: float
    q25_T: float
    q75_T: float
    frac_hit_cap: float
    error_rate: float
    truth: float
    truth_above_tau: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_out_dir() -> Path:
    return _repo_root() / "outputs" / "m3"


def _rectified_stream(
    y: np.ndarray,
    f: np.ndarray,
    order: np.ndarray,
    *,
    lam: float = LAM,
) -> np.ndarray:
    f_bar = float(f.mean())
    idx = order
    return y[idx] - lam * (f[idx] - f_bar)


def run_one_stream(
    y: np.ndarray,
    f: np.ndarray,
    *,
    seed: int,
    t_max: int = T_MAX,
    alpha: float = ALPHA,
    tau: float = TAU,
) -> dict:
    rng = np.random.default_rng(seed)
    order = rng.permutation(y.size)
    z = _rectified_stream(y, f, order)
    z_min, z_max = ppi_rectifier_bounds(lam=LAM)
    u = affine_to_unit(z[:t_max], z_min=z_min, z_max=z_max)

    cs = HedgedBettingCS(alpha=alpha, grid=1001)
    states = []
    for x in u:
        states.append(cs.update(float(x)))

    # Map CS back to original scale for threshold comparisons.
    # Threshold tau on theta-scale maps to unit interval via the same affine map.
    tau_u = float(affine_to_unit(np.array([tau]), z_min=z_min, z_max=z_max)[0])

    t_clear = first_clear_above(states, threshold=tau_u)
    t_excl = first_excludes(states, threshold=tau_u)

    # Final CS on original scale (for diagnostics).
    final = states[-1]
    return {
        "t_clear_above": t_clear if t_clear is not None else t_max,
        "cleared_above": t_clear is not None,
        "t_excludes": t_excl if t_excl is not None else t_max,
        "excluded": t_excl is not None,
        "final_lower_u": final.lower,
        "final_upper_u": final.upper,
        "final_lower": affine_from_unit(final.lower, z_min=z_min, z_max=z_max),
        "final_upper": affine_from_unit(final.upper, z_min=z_min, z_max=z_max),
        "decision_above": bool(
            t_clear is not None
            and affine_from_unit(
                states[t_clear - 1].lower, z_min=z_min, z_max=z_max
            )
            > tau
        ),
    }


def run_stopping(
    *,
    R: int = 500,
    smoke: bool = False,
    t_max: int = T_MAX,
) -> tuple[pd.DataFrame, dict]:
    _, human, load_report = load_toxicchat(strict=True)
    y = human["toxicity"].astype(float).to_numpy()
    f = OpenAIModerationMaxJudge().score(human)
    truth = float(y.mean())
    truth_above = truth > TAU

    records = []
    for r in range(R):
        out = run_one_stream(y, f, seed=stopping_seed(r), t_max=t_max)
        out["r"] = r
        records.append(out)
    df = pd.DataFrame(records)

    # Error for clear-above: stopped claiming theta > tau when truth <= tau.
    # If truth > tau, a "error" is failing to eventually clear by t_max is not
    # typed as error under the locked plan; only false claims count.
    if truth_above:
        # False decision: cleared above incorrectly cannot happen if truth>tau
        # for a valid CS, but numerical issues possible; also count clears that
        # somehow have lower <= tau (should not). Primary error: stop with
        # clear-above when the mapped lower does not exceed tau.
        err_clear = (~df["decision_above"]) & df["cleared_above"]
        # Locked definition: error = stop claiming theta > tau when truth <= tau.
        # When truth > tau, false-positive rate for that claim is 0 by definition
        # of the estimand; report empirical rate of contradictory clears as 0
        # unless decision_above is false on a clear (numerical).
        error_rate_clear = float(err_clear.mean())
    else:
        error_rate_clear = float(df["cleared_above"].mean())

    # Two-sided exclude: error if we exclude tau on the wrong side of truth.
    # If truth > tau, error when final decision says upper < tau (claim below).
    # If truth < tau, error when lower > tau.
    if truth_above:
        # Runs that exclude by going below tau are errors.
        wrong = []
        for _, row in df.iterrows():
            if not row["excluded"]:
                wrong.append(False)
            else:
                wrong.append(row["final_upper"] < TAU)
        error_rate_excl = float(np.mean(wrong))
    else:
        wrong = []
        for _, row in df.iterrows():
            if not row["excluded"]:
                wrong.append(False)
            else:
                wrong.append(row["final_lower"] > TAU)
        error_rate_excl = float(np.mean(wrong))

    summaries = [
        StoppingSummary(
            rule="clear_above",
            R=R,
            alpha=ALPHA,
            tau=TAU,
            t_max=t_max,
            median_T=float(df["t_clear_above"].median()),
            mean_T=float(df["t_clear_above"].mean()),
            q25_T=float(df["t_clear_above"].quantile(0.25)),
            q75_T=float(df["t_clear_above"].quantile(0.75)),
            frac_hit_cap=float((~df["cleared_above"]).mean()),
            error_rate=error_rate_clear,
            truth=truth,
            truth_above_tau=truth_above,
        ),
        StoppingSummary(
            rule="excludes_tau",
            R=R,
            alpha=ALPHA,
            tau=TAU,
            t_max=t_max,
            median_T=float(df["t_excludes"].median()),
            mean_T=float(df["t_excludes"].mean()),
            q25_T=float(df["t_excludes"].quantile(0.25)),
            q75_T=float(df["t_excludes"].quantile(0.75)),
            frac_hit_cap=float((~df["excluded"]).mean()),
            error_rate=error_rate_excl,
            truth=truth,
            truth_above_tau=truth_above,
        ),
    ]

    meta = {
        "seed_root": SEED_ROOT,
        "R": R,
        "smoke": smoke,
        "tau": TAU,
        "t_max": t_max,
        "alpha": ALPHA,
        "lam": LAM,
        "truth_toxicity": truth,
        "judge": OpenAIModerationMaxJudge.name,
        "load_report": json.loads(load_report.to_json()),
        "summaries": [asdict(s) for s in summaries],
        "headline": {
            "rule": "clear_above",
            "median_stopping_time": summaries[0].median_T,
            "error_rate": summaries[0].error_rate,
            "frac_hit_cap": summaries[0].frac_hit_cap,
            "truth": truth,
            "tau": TAU,
        },
    }
    return df, meta


def run_and_save(
    *,
    R: int = 500,
    smoke: bool = False,
    out_dir: Path | None = None,
) -> Path:
    out = out_dir or default_out_dir()
    out.mkdir(parents=True, exist_ok=True)
    df, meta = run_stopping(R=R, smoke=smoke)
    tag = "smoke" if smoke or R < 500 else "full"
    runs_path = out / f"stopping_runs_{tag}.csv"
    meta_path = out / f"stopping_meta_{tag}.json"
    df.to_csv(runs_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return runs_path
