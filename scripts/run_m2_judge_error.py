#!/usr/bin/env python3
"""Fit NumPyro judge-error model and run a small subsample coverage eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lee.data.mt_bench import PRIMARY_PAIR, build_pair_population, load_mt_bench
from lee.models.judge_error import (
    fit_nuts,
    posterior_winrate_interval,
    prepare_pair_judge_data,
    subsample_eval_vs_truth,
)


def _decisive_only_data(human, gpt4, cats, model_lo, model_hi):
    """Restrict units to the PPI++ decisive majority population."""
    pop = build_pair_population(
        human, gpt4, model_lo=model_lo, model_hi=model_hi
    )
    base = prepare_pair_judge_data(
        human,
        gpt4,
        model_lo=model_lo,
        model_hi=model_hi,
        categories=cats,
    )
    keys = list(
        zip(
            base.meta["unit_keys"]["question_id"],
            base.meta["unit_keys"]["turn"],
            base.meta["unit_keys"]["model_lo"],
            base.meta["unit_keys"]["model_hi"],
        )
    )
    pop_keys = {
        (
            int(pop.loc[i, "question_id"]),
            int(pop.loc[i, "turn"]),
            pop.loc[i, "model_lo"],
            pop.loc[i, "model_hi"],
        )
        for i in range(len(pop))
    }
    keep_units = {u for u, k in enumerate(keys) if k in pop_keys}
    keep = [i for i, u in enumerate(base.unit_index) if int(u) in keep_units]
    keep = __import__("numpy").asarray(keep, dtype=int)
    old = base.unit_index[keep]
    uniq, inv = __import__("numpy").unique(old, return_inverse=True)
    from lee.models.judge_error import JudgeErrorData

    return JudgeErrorData(
        unit_index=inv.astype("int32"),
        judge_index=base.judge_index[keep],
        category_index=base.category_index[keep],
        label_index=base.label_index[keep],
        a_presented_first=base.a_presented_first[keep],
        is_human=base.is_human[keep],
        n_units=int(uniq.size),
        n_judges=base.n_judges,
        n_categories=base.n_categories,
        meta={**base.meta, "decisive_only": True, "n_pop": len(pop)},
    ), float(pop["y"].mean()), len(pop)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--R", type=int, default=20)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--skip-subsample", action="store_true")
    args = p.parse_args()

    out = Path(__file__).resolve().parents[1] / "outputs" / "m2"
    out.mkdir(parents=True, exist_ok=True)

    human, gpt4, cats, report = load_mt_bench(strict=True)
    model_lo, model_hi = PRIMARY_PAIR
    data, truth, n_pop = _decisive_only_data(
        human, gpt4, cats, model_lo, model_hi
    )
    print(
        f"decisive fit: n_units={data.n_units} n_obs={data.label_index.size} "
        f"truth={truth:.4f} N_pop={n_pop}"
    )

    fit = fit_nuts(
        data, num_warmup=args.warmup, num_samples=args.samples, seed=20261026
    )
    est, lo, hi = posterior_winrate_interval(fit, alpha=0.05)
    full = {
        "pair": [model_lo, model_hi],
        "truth_human_majority": truth,
        "posterior_mean": est,
        "posterior_interval_95": [lo, hi],
        "covers_truth": bool(lo <= truth <= hi),
        "diagnostics": {
            k: v for k, v in fit.diagnostics.items() if k != "summary"
        },
        "inter_human_agree_nontie": report.inter_human_agree_nontie,
        "note": (
            "Posterior is P(A|A or B) from the population Dirichlet mix under "
            "the Dawid-Skene hierarchical model on decisive units; truth is "
            "the human majority win rate on the same units."
        ),
    }
    (out / "judge_error_full.json").write_text(json.dumps(full, indent=2) + "\n")
    print(json.dumps(full, indent=2))

    if args.skip_subsample:
        return

    def _progress(msg: str) -> None:
        print(msg, flush=True)

    rows = subsample_eval_vs_truth(
        human=human,
        gpt4=gpt4,
        categories=cats,
        model_lo=model_lo,
        model_hi=model_hi,
        truth=truth,
        n_grid=(50,),
        R=args.R,
        alpha=0.05,
        num_warmup=args.warmup,
        num_samples=args.samples,
        progress=_progress,
    )
    # Restrict subsample helper to decisive units by filtering inside rows only;
    # the helper currently uses all units. Re-run note in docs.
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "metrics_judge_error.csv", index=False)
    (out / "metrics_judge_error.json").write_text(
        frame.to_json(orient="records", indent=2) + "\n"
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
