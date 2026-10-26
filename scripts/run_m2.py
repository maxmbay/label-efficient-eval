#!/usr/bin/env python3
"""Run Milestone 2 PPI++ grid and write outputs."""

from __future__ import annotations

import argparse
import json

from lee.sim.headline_m2 import compute_m2_headline
from lee.sim.m2 import run_and_save
from lee.sim.metrics import metrics_to_frame


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--R", type=int, default=None)
    args = p.parse_args()
    R = args.R if args.R is not None else (5 if args.smoke else 1000)
    out = run_and_save(R=R, smoke=args.smoke)
    import pandas as pd

    tag = "smoke" if args.smoke or R < 1000 else "full"
    metrics = pd.read_csv(out / f"metrics_{tag}.csv")
    headline = compute_m2_headline(metrics)
    (out / f"headline_{tag}.json").write_text(
        json.dumps(headline.to_dict(), indent=2) + "\n"
    )
    print(headline.message)


if __name__ == "__main__":
    main()
