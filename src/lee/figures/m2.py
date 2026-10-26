"""Milestone 2 figure: mean width and coverage vs n for the primary pair."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_m2(
    metrics: pd.DataFrame,
    *,
    alpha: float = 0.05,
    estimators: list[str] | None = None,
    out_path: Path | None = None,
    title_suffix: str = "",
) -> Path:
    estimators = estimators or [
        "classical_clt",
        "classical_clustered",
        "ppi",
        "ppi_pp",
        "judge_only",
        "judge_error_nuts",
    ]
    sub = metrics[
        (metrics["alpha"] == alpha) & (metrics["estimator"].isin(estimators))
    ].copy()
    if sub.empty:
        raise ValueError("No rows to plot")

    fig, (ax_w, ax_c) = plt.subplots(
        2, 1, figsize=(7.5, 6.5), sharex=True, constrained_layout=True
    )
    for estimator, g in sub.groupby("estimator"):
        g = g.sort_values("n")
        ax_w.plot(g["n"], g["mean_width"], marker="o", label=estimator)
        ax_c.plot(g["n"], g["coverage"], marker="o", label=estimator)
        if "coverage_se" in g.columns:
            ax_c.fill_between(
                g["n"],
                g["coverage"] - 1.96 * g["coverage_se"],
                g["coverage"] + 1.96 * g["coverage_se"],
                alpha=0.12,
            )

    ax_w.set_ylabel("Mean interval width")
    ax_w.set_xscale("log")
    ax_w.legend(fontsize=8, loc="best")
    estimand = str(sub["estimand"].iloc[0]) if "estimand" in sub.columns else "winrate"
    ax_w.set_title(
        f"M2 {estimand}, nominal {(1 - alpha):.0%} intervals{title_suffix}"
    )

    target = 1.0 - alpha
    ax_c.axhline(target, color="black", linestyle="--", linewidth=1)
    ax_c.set_ylabel("Empirical coverage")
    ax_c.set_xlabel("Labeled sample size n")
    ax_c.set_xscale("log")
    ax_c.set_ylim(0.0, 1.05)

    out = out_path or Path("outputs/m2/figure_m2.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    out_dir = root / "outputs" / "m2"
    csv = out_dir / "metrics_full.csv"
    if not csv.exists():
        csv = out_dir / "metrics_smoke.csv"
    metrics = pd.read_csv(csv)
    # Optionally merge Bayes arm if present
    bayes = out_dir / "metrics_judge_error.csv"
    if bayes.exists():
        metrics = pd.concat([metrics, pd.read_csv(bayes)], ignore_index=True)
    path = plot_m2(metrics, out_path=out_dir / "figure_m2.png")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
