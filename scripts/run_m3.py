#!/usr/bin/env python3
"""Run all Milestone 3 studies (allocation, stopping, Criteo) then figures."""

from __future__ import annotations

import argparse

from lee.figures.m3 import plot_m3
from lee.sim import m3_allocation, m3_criteo, m3_stopping


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--R", type=int, default=None, help="Override R for alloc/stop.")
    p.add_argument("--skip-criteo", action="store_true")
    p.add_argument("--criteo-full", action="store_true")
    args = p.parse_args()

    smoke = args.smoke
    R_alloc = args.R if args.R is not None else (5 if smoke else 500)
    R_stop = args.R if args.R is not None else (5 if smoke else 500)

    print("=== allocation ===")
    print(m3_allocation.run_and_save(R=R_alloc, smoke=smoke or R_alloc < 500))
    print("=== stopping ===")
    print(m3_stopping.run_and_save(R=R_stop, smoke=smoke or R_stop < 500))
    if not args.skip_criteo:
        print("=== criteo ===")
        print(
            m3_criteo.run_and_save(
                smoke=smoke,
                full=args.criteo_full,
            )
        )
    tag = "smoke" if smoke or R_alloc < 500 else "full"
    print("=== figure ===")
    print(plot_m3(tag=tag))


if __name__ == "__main__":
    main()
