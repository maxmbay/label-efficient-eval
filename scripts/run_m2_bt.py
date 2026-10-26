#!/usr/bin/env python3
"""Fit Bradley-Terry strengths on MT-Bench human majority (non-tie)."""

from __future__ import annotations

import json
from pathlib import Path

from lee.data.mt_bench import collapse_human_majority, load_mt_bench
from lee.estimators.bradley_terry import fit_bradley_terry


def main() -> None:
    human, _gpt4, _cats, _report = load_mt_bench(strict=True)
    collapsed = collapse_human_majority(human)
    result = fit_bradley_terry(
        collapsed,
        model_a_col="model_lo",
        model_b_col="model_hi",
        winner_col="majority_winner",
        question_col="question_id",
        n_bootstrap=500,
        seed=20261026,
    )
    out = Path(__file__).resolve().parents[1] / "outputs" / "m2"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "bradley_terry.json"
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    print(f"Wrote {path}")
    print("strengths (mean-zero):")
    for m, s in sorted(result.strengths.items(), key=lambda kv: -kv[1]):
        print(f"  {m:20s}  {s:+.3f}  (SE {result.se[m]:.3f})")


if __name__ == "__main__":
    main()
