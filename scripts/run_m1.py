#!/usr/bin/env python3
"""CLI entry for the Milestone 1 simulation grid."""

from __future__ import annotations

import argparse

from lee.sim.m1 import DEFAULT_N_GRID, run_and_save


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--R",
        type=int,
        default=1000,
        help="Monte Carlo replicates per (n, estimand, judge). Default 1000.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke run: R=5 on the full n grid (or override with --R).",
    )
    p.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=None,
        help=f"Label budgets. Default: {list(DEFAULT_N_GRID)}",
    )
    args = p.parse_args()
    R = 5 if args.smoke and args.R == 1000 else args.R
    n_grid = tuple(args.n) if args.n is not None else None
    run_and_save(R=R, n_grid=n_grid, smoke=args.smoke or R < 1000)


if __name__ == "__main__":
    main()
