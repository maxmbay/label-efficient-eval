#!/usr/bin/env python3
"""CLI: Milestone 3 Criteo CUPAC bridge."""

from __future__ import annotations

import argparse

from lee.data.criteo import DEFAULT_SUBSAMPLE_N
from lee.sim.m3_criteo import run_and_save


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true", help="Load entire Criteo CSV.")
    p.add_argument(
        "--subsample-n",
        type=int,
        default=DEFAULT_SUBSAMPLE_N,
        help="Row count for local subsample (ignored with --full).",
    )
    args = p.parse_args()
    path = run_and_save(
        smoke=args.smoke,
        subsample_n=None if args.full else args.subsample_n,
        full=args.full,
    )
    print(path)


if __name__ == "__main__":
    main()
