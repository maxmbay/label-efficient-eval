"""Milestone 1 figure: mean width and coverage vs n."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_m1(
    metrics: pd.DataFrame,
    *,
    estimand: str = "toxicity",
    alpha: float = 0.05,
    judges: list[str] | None = None,
    estimators: list[str] | None = None,
    out_path: Path | None = None,
) -> Path:
    """Width (top) and coverage (bottom) vs n for one estimand / alpha."""
    estimators = estimators or ["classical_clt", "ppi", "ppi_pp", "judge_only"]
    sub = metrics[
        (metrics["estimand"] == estimand)
        & (metrics["alpha"] == alpha)
        & (metrics["estimator"].isin(estimators))
    ].copy()
    if judges is not None:
        sub = sub[sub["judge"].isin(judges)]
    if sub.empty:
        raise ValueError("No rows to plot for the requested slice")

    fig, (ax_w, ax_c) = plt.subplots(
        2, 1, figsize=(7.5, 6.5), sharex=True, constrained_layout=True
    )
    for (judge, estimator), g in sub.groupby(["judge", "estimator"]):
        g = g.sort_values("n")
        label = f"{estimator} | {judge}"
        ax_w.plot(g["n"], g["mean_width"], marker="o", label=label)
        ax_c.plot(g["n"], g["coverage"], marker="o", label=label)
        ax_c.fill_between(
            g["n"],
            g["coverage"] - 1.96 * g["coverage_se"],
            g["coverage"] + 1.96 * g["coverage_se"],
            alpha=0.15,
        )

    ax_w.set_ylabel("Mean interval width")
    ax_w.set_xscale("log")
    ax_w.legend(fontsize=7, loc="best")
    ax_w.set_title(f"M1 {estimand}, nominal {(1 - alpha):.0%} intervals")

    target = 1.0 - alpha
    ax_c.axhline(target, color="black", linestyle="--", linewidth=1)
    ax_c.set_ylabel("Empirical coverage")
    ax_c.set_xlabel("Labeled sample size n")
    ax_c.set_xscale("log")
    ax_c.set_ylim(0.0, 1.05)

    out = out_path or Path("outputs/m1/figure_m1.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    out_dir = root / "outputs" / "m1"
    # Prefer full metrics; fall back to smoke.
    csv = out_dir / "metrics_full.csv"
    if not csv.exists():
        csv = out_dir / "metrics_smoke.csv"
    if not csv.exists():
        raise SystemExit(
            f"No metrics CSV under {out_dir}. Run `make m1` or `make m1-smoke` first."
        )
    metrics = pd.read_csv(csv)
    path = plot_m1(metrics, out_path=out_dir / "figure_m1.png")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
