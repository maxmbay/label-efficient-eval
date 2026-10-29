"""On-disk cache for judge soft scores (no raw prompts committed)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def judge_cache_dir() -> Path:
    d = _repo_root() / "data" / "cache" / "judges"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_path(judge_name: str) -> Path:
    return judge_cache_dir() / f"{judge_name}.parquet"


def meta_path(judge_name: str) -> Path:
    return judge_cache_dir() / f"{judge_name}.meta.json"


def save_scores(
    judge_name: str,
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    meta: dict | None = None,
) -> Path:
    """Persist ``conv_id`` + score (+ split if present)."""
    scores = np.asarray(scores, dtype=float).ravel()
    if len(frame) != scores.size:
        raise ValueError("frame/scores length mismatch")
    out = pd.DataFrame(
        {
            "conv_id": frame["conv_id"].astype(str).to_numpy(),
            "score": scores,
        }
    )
    if "split" in frame.columns:
        out["split"] = frame["split"].astype(str).to_numpy()
    path = cache_path(judge_name)
    out.to_parquet(path, index=False)
    payload = {"judge": judge_name, "n": int(scores.size), **(meta or {})}
    meta_path(judge_name).write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_scores(judge_name: str) -> pd.DataFrame | None:
    path = cache_path(judge_name)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def align_scores(population: pd.DataFrame, cached: pd.DataFrame) -> np.ndarray:
    """Return score vector aligned to ``population`` row order via conv_id."""
    merged = population[["conv_id"]].merge(
        cached[["conv_id", "score"]], on="conv_id", how="left", validate="one_to_one"
    )
    if merged["score"].isna().any():
        n_miss = int(merged["score"].isna().sum())
        raise ValueError(f"Cached scores missing {n_miss} population rows")
    return merged["score"].to_numpy(dtype=float)
