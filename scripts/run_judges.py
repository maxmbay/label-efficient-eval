#!/usr/bin/env python3
"""Score the analysis population with a judge and cache soft scores."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from lee.data.toxicchat import load_toxicchat
from lee.judges.auth import JudgeAuthError, LLAMA_GUARD_SETUP, require_modal
from lee.judges.cache import save_scores
from lee.judges.crossfit import maybe_crossfit
from lee.judges.llama_guard import LLAMA_GUARD_3_1B, LlamaGuard3Judge
from lee.judges.open_it import OpenITJudge
from lee.judges.shieldgemma import ShieldGemmaJudge


def _score_modal_llama(texts: list[str], model_id: str) -> np.ndarray:
    require_modal()
    import modal

    score_fn = modal.Function.from_name("lee-m1-judges", "score_texts_llama_guard")
    # Chunk to keep payload reasonable.
    out: list[float] = []
    chunk = 256
    for i in range(0, len(texts), chunk):
        part = score_fn.remote(texts[i : i + chunk], model_id=model_id)
        out.extend(part)
        print(f"modal llama_guard {i + len(part)}/{len(texts)}", flush=True)
    return np.asarray(out, dtype=float)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--judge",
        required=True,
        choices=[
            "llama_guard3_1b",
            "shieldgemma_2b",
            "open_it_toxicity",
            "open_it_jailbreaking",
        ],
    )
    p.add_argument(
        "--backend",
        choices=["local", "modal"],
        default="local",
        help="local = MPS/CPU on this machine; modal = personal Modal GPU",
    )
    p.add_argument(
        "--calibrate",
        action="store_true",
        help="Cross-fit Platt scaling using human labels (train/test).",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="Debug: score first N rows")
    args = p.parse_args()

    _all, population, report = load_toxicchat(strict=True)
    if args.limit is not None:
        population = population.iloc[: args.limit].copy()
    print(report.to_json(), flush=True)

    label_col = (
        "jailbreaking" if "jailbreak" in args.judge else "toxicity"
    )
    try:
        if args.judge == "llama_guard3_1b":
            if args.backend == "modal":
                raw = _score_modal_llama(
                    population["user_input"].astype(str).tolist(),
                    LLAMA_GUARD_3_1B,
                )
            else:
                raw = LlamaGuard3Judge(batch_size=args.batch_size).score(population)
            name = "llama_guard3_1b"
        elif args.judge == "shieldgemma_2b":
            if args.backend == "modal":
                raise JudgeAuthError(
                    "ShieldGemma Modal path not wired yet; use --backend local "
                    "after HF license accept, or extend modal_apps/judges.py."
                )
            raw = ShieldGemmaJudge(batch_size=args.batch_size).score(population)
            name = "shieldgemma_2b"
        elif args.judge == "open_it_toxicity":
            raw = OpenITJudge(
                estimand="toxicity", batch_size=args.batch_size
            ).score(population)
            name = "open_it_toxicity"
        else:
            raw = OpenITJudge(
                estimand="jailbreaking", batch_size=args.batch_size
            ).score(population)
            name = "open_it_jailbreaking"

        scores, did_cal = maybe_crossfit(
            population, raw, label_col=label_col, calibrate=args.calibrate
        )
        if did_cal:
            name = f"{name}_crossfit"
        path = save_scores(
            name,
            population,
            scores,
            meta={
                "backend": args.backend,
                "calibrated": did_cal,
                "label_col": label_col,
                "n": int(len(scores)),
                "mean": float(np.mean(scores)),
            },
        )
        print(json.dumps({"saved": str(path), "name": name, "mean": float(scores.mean())}))
    except JudgeAuthError as exc:
        print(str(exc), file=sys.stderr)
        print("\n" + LLAMA_GUARD_SETUP, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
