#!/usr/bin/env python3
"""CLI: Milestone 3 allocation grid (active vs uniform AIPW)."""

from __future__ import annotations

import argparse

from lee.sim.m3_allocation import DEFAULT_N_GRID, run_and_save


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--R", type=int, default=500)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--n", type=int, nargs="+", default=None)
    args = p.parse_args()
    R = 5 if args.smoke and args.R == 500 else args.R
    n_grid = tuple(args.n) if args.n is not None else DEFAULT_N_GRID
    path = run_and_save(R=R, n_grid=n_grid, smoke=args.smoke or R < 500)
    print(path)


if __name__ == "__main__":
    main()
