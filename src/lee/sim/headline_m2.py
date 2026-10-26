"""Honest M2 headline from locked definition (do not invent savings)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class M2Headline:
    n: int
    alpha: float
    judges_meeting_coverage: tuple[str, ...]
    best_judge: str | None
    w_ppi_pp: float | None
    n_classical: int | None
    labels_saved: float | None
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def compute_m2_headline(
    metrics: pd.DataFrame,
    *,
    n: int = 50,
    alpha: float = 0.05,
    ppi_estimator: str = "ppi_pp",
    classical_estimator: str = "classical_clustered",
) -> M2Headline:
    """Apply analysis_plan.md M2 §8. Never invent a savings percentage."""
    sub = metrics[(metrics["alpha"] == alpha) & (metrics["n"] == n)].copy()
    if sub.empty:
        return M2Headline(
            n=n,
            alpha=alpha,
            judges_meeting_coverage=(),
            best_judge=None,
            w_ppi_pp=None,
            n_classical=None,
            labels_saved=None,
            message=(
                f"No metrics at n={n}, alpha={alpha}; cannot form an M2 headline."
            ),
        )

    # Coverage within one Monte Carlo SE of 95%
    target = 1.0 - alpha
    meeting = []
    for _, row in sub.iterrows():
        if row["estimator"] != ppi_estimator:
            continue
        se = float(row["coverage_se"])
        cov = float(row["coverage"])
        if se == 0.0:
            ok = cov == target
        else:
            ok = abs(cov - target) <= se
        if ok:
            meeting.append(str(row["judge"]))

    ppi_rows = sub[sub["estimator"] == ppi_estimator]
    if not meeting:
        # Report failure cleanly
        detail = "; ".join(
            f"{r.judge}: coverage={r.coverage:.3f} (SE={r.coverage_se:.3f}, "
            f"width={r.mean_width:.3f})"
            for _, r in ppi_rows.iterrows()
        )
        return M2Headline(
            n=n,
            alpha=alpha,
            judges_meeting_coverage=(),
            best_judge=None,
            w_ppi_pp=None,
            n_classical=None,
            labels_saved=None,
            message=(
                f"No judge meets nominal {target:.0%} coverage within Monte Carlo "
                f"SE at n={n} under {ppi_estimator}. Do not report a labels-saved "
                f"percentage. Detail: {detail}"
            ),
        )

    # Best among meeting by mean width
    cand = ppi_rows[ppi_rows["judge"].isin(meeting)].sort_values("mean_width")
    best = cand.iloc[0]
    w = float(best["mean_width"])
    # classical curve: smallest n with mean_width <= w
    clas = metrics[
        (metrics["alpha"] == alpha)
        & (metrics["estimator"] == classical_estimator)
        & (metrics["judge"] == best["judge"])
    ].sort_values("n")
    n_classical = None
    for _, row in clas.iterrows():
        if float(row["mean_width"]) <= w:
            n_classical = int(row["n"])
            break
    if n_classical is None:
        msg = (
            f"Judge {best['judge']} meets coverage at n={n} with PPI++ width "
            f"{w:.4f}, but no classical n on the grid matches that width."
        )
        saved = None
    else:
        saved = 1.0 - (n / n_classical)
        msg = (
            f"At n={n}, judge {best['judge']} PPI++ mean width {w:.4f} with "
            f"coverage {best['coverage']:.3f}; classical_clustered needs "
            f"n={n_classical} for equal-or-better width "
            f"(labels saved = {saved:.1%})."
        )
    return M2Headline(
        n=n,
        alpha=alpha,
        judges_meeting_coverage=tuple(meeting),
        best_judge=str(best["judge"]),
        w_ppi_pp=w,
        n_classical=n_classical,
        labels_saved=saved,
        message=msg,
    )
