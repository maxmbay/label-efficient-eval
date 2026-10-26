"""Locked Milestone 1 headline from simulation metrics.

Definition (analysis_plan.md section 6): at n=200, among judges whose
empirical 95% coverage is within one Monte Carlo SE of 0.95 under PPI++,
take the narrowest mean width; find the smallest classical n on the grid
with mean width at most that width; report 1 - n/n_classical. If none
qualify, report failure (no savings percentage).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd

HEADLINE_N = 200
NOMINAL_COVERAGE = 0.95
ALPHA = 0.05


def format_labels_saved_pct(labels_saved: float, *, decimals: int = 1) -> str:
    """Format a fraction as a percent without rounding up."""
    scale = 10**decimals
    floored = math.floor(labels_saved * 100 * scale) / scale
    return f"{floored:.{decimals}f}%"


@dataclass(frozen=True)
class HeadlineResult:
    estimand: str
    eligible: bool
    judge: str | None
    w_ppi_pp: float | None
    n_classical: int | None
    labels_saved: float | None
    message: str
    coverage: float | None = None
    coverage_se: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def compute_headline(
    metrics: pd.DataFrame,
    *,
    estimand: str = "toxicity",
    n: int = HEADLINE_N,
    alpha: float = ALPHA,
) -> HeadlineResult:
    sub = metrics[(metrics["estimand"] == estimand) & (metrics["alpha"] == alpha)]
    pp = sub[(sub["estimator"] == "ppi_pp") & (sub["n"] == n)]
    if pp.empty:
        return HeadlineResult(
            estimand=estimand,
            eligible=False,
            judge=None,
            w_ppi_pp=None,
            n_classical=None,
            labels_saved=None,
            message=f"No PPI++ rows at n={n} for {estimand}",
        )

    eligible_rows = []
    detail = []
    for _, row in pp.iterrows():
        within = abs(float(row["coverage"]) - NOMINAL_COVERAGE) <= float(
            row["coverage_se"]
        ) + 1e-15
        detail.append(
            f"{row['judge']}: coverage={row['coverage']:.4f}±{row['coverage_se']:.4f} "
            f"eligible={within}"
        )
        if within:
            eligible_rows.append(row)

    if not eligible_rows:
        return HeadlineResult(
            estimand=estimand,
            eligible=False,
            judge=None,
            w_ppi_pp=None,
            n_classical=None,
            labels_saved=None,
            message=(
                "No judge met the coverage criterion "
                f"(|coverage - {NOMINAL_COVERAGE}| <= MC SE) under PPI++ at "
                f"n={n}. "
                + "; ".join(detail)
            ),
        )

    best = min(eligible_rows, key=lambda r: float(r["mean_width"]))
    w = float(best["mean_width"])
    cl = sub[
        (sub["estimator"] == "classical_clt") & (sub["judge"] == best["judge"])
    ].sort_values("n")
    n_classical = None
    for _, row in cl.iterrows():
        if float(row["mean_width"]) <= w:
            n_classical = int(row["n"])
            break
    if n_classical is None:
        return HeadlineResult(
            estimand=estimand,
            eligible=True,
            judge=str(best["judge"]),
            w_ppi_pp=w,
            n_classical=None,
            labels_saved=None,
            coverage=float(best["coverage"]),
            coverage_se=float(best["coverage_se"]),
            message=(
                f"Best eligible judge {best['judge']} has w={w:.6f}, but no "
                "classical n on the grid has mean width <= that."
            ),
        )
    saved = 1.0 - n / n_classical
    return HeadlineResult(
        estimand=estimand,
        eligible=True,
        judge=str(best["judge"]),
        w_ppi_pp=w,
        n_classical=n_classical,
        labels_saved=saved,
        coverage=float(best["coverage"]),
        coverage_se=float(best["coverage_se"]),
        message=(
            f"labels_saved = 1 - {n}/{n_classical} = {saved:.6f} "
            f"(judge={best['judge']}, w_ppi++={w:.6f})"
        ),
    )
