#!/usr/bin/env python3
"""CLI: Milestone 3 stopping-time simulation (betting CS)."""

from __future__ import annotations

import argparse

from lee.sim.m3_stopping import run_and_save


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--R", type=int, default=500)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    R = 5 if args.smoke and args.R == 500 else args.R
    path = run_and_save(R=R, smoke=args.smoke or R < 500)
    print(path)


if __name__ == "__main__":
    main()
