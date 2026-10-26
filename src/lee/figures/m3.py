"""Milestone 3 figures: allocation width, stopping times, Criteo variance."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_out_dir() -> Path:
    return _repo_root() / "outputs" / "m3"


def plot_m3(
    *,
    out_dir: Path | None = None,
    tag: str = "full",
) -> Path:
    out = out_dir or default_out_dir()
    out.mkdir(parents=True, exist_ok=True)

    alloc_path = out / f"allocation_{tag}.csv"
    stop_meta = out / f"stopping_meta_{tag}.json"
    stop_runs = out / f"stopping_runs_{tag}.csv"
    criteo_path = out / f"criteo_{tag}.json"

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))

    # Panel 1: allocation mean width vs n
    ax = axes[0]
    if alloc_path.exists():
        adf = pd.read_csv(alloc_path)
        adf = adf.loc[adf["alpha"] == 0.05]
        for method, style in (("uniform", "-o"), ("active", "-s")):
            sub = adf.loc[adf["method"] == method].sort_values("n_budget")
            ax.plot(
                sub["n_budget"],
                sub["mean_width"],
                style,
                label=method,
                markersize=4,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Expected label budget $n_b$")
        ax.set_ylabel("Mean 95% width")
        ax.set_title("Allocation (ToxicChat tox)")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "missing allocation", ha="center", va="center")
        ax.set_axis_off()

    # Panel 2: stopping time histogram
    ax = axes[1]
    if stop_runs.exists():
        sdf = pd.read_csv(stop_runs)
        ax.hist(sdf["t_clear_above"], bins=30, color="#4c566a", edgecolor="white")
        if stop_meta.exists():
            meta = json.loads(stop_meta.read_text())
            med = meta["headline"]["median_stopping_time"]
            ax.axvline(med, color="#bf616a", linestyle="--", label=f"median={med:.0f}")
            ax.legend(frameon=False)
        ax.set_xlabel("Stopping time $T$ (clear-above)")
        ax.set_ylabel("Count")
        ax.set_title("Stopping (betting CS)")
    else:
        ax.text(0.5, 0.5, "missing stopping", ha="center", va="center")
        ax.set_axis_off()

    # Panel 3: Criteo variance bars
    ax = axes[2]
    if criteo_path.exists():
        c = json.loads(criteo_path.read_text())
        methods = ["DIM", "CUPAC"]
        vars_ = [c["dim"]["var"], c["cupac"]["var"]]
        ax.bar(methods, vars_, color=["#5e81ac", "#a3be8c"])
        ax.set_ylabel("Asymptotic variance")
        factor = c["headline"]["variance_reduction_factor"]
        ax.set_title(f"Criteo ATE on visit (ratio={factor:.2f})")
        n_used = c["headline"]["n_rows_used"]
        ax.text(
            0.5,
            -0.18,
            f"n={n_used:,}"
            + (" subsample" if c["headline"]["subsampled"] else " full"),
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
        )
    else:
        ax.text(0.5, 0.5, "missing criteo", ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout()
    dest = out / f"figure_m3_{tag}.png"
    fig.savefig(dest, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Also write a stable name for README when tag=full.
    if tag == "full":
        stable = out / "figure_m3.png"
        stable.write_bytes(dest.read_bytes())
    return dest


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="full", choices=("full", "smoke"))
    args = p.parse_args()
    path = plot_m3(tag=args.tag)
    print(path)


if __name__ == "__main__":
    main()
