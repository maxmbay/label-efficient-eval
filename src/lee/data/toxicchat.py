"""Load and verify lmsys/toxic-chat (toxicchat0124).

Population for Milestone 1: train+test pooled, human_annotation == True.
Raw data is cached under data/raw/ (gitignored); never committed.
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# Expected sizes from the Hugging Face dataset card for toxicchat0124
# (https://huggingface.co/datasets/lmsys/toxic-chat).
EXPECTED_N_TOTAL = 10_165
EXPECTED_N_HUMAN = 5_654
EXPECTED_TRAIN = 5_082
EXPECTED_TEST = 5_083

# Paper (Lin et al., arXiv 2310.17389) overall rates on the released set at
# publication time. The 0124 card reports slightly higher rates after fixes.
PAPER_TOX_RATE = 0.0710
PAPER_JB_RATE = 0.0175
# First-stage process note in the paper: ~61% of prompts were never shown to
# humans (Perspective score below ~0.037). The released 0124 set has more
# human labels than that first-stage fraction (see LoadReport).
PAPER_FIRST_STAGE_UNLABELED_FRAC = 0.61

REQUIRED_COLUMNS = (
    "conv_id",
    "user_input",
    "model_output",
    "human_annotation",
    "toxicity",
    "jailbreaking",
    "openai_moderation",
)


@dataclass(frozen=True)
class LoadReport:
    """Verification summary written beside the cached table."""

    n_train: int
    n_test: int
    n_total: int
    n_human: int
    n_auto: int
    human_frac: float
    tox_rate_all: float
    jb_rate_all: float
    tox_rate_human: float
    jb_rate_human: float
    paper_tox_rate: float
    paper_jb_rate: float
    paper_first_stage_unlabeled_frac: float
    hf_card_n_total: int
    hf_card_n_human: int
    semantics_ok: bool
    counts_match_hf_card: bool
    notes: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class ToxicChatLoadError(RuntimeError):
    """Raised when human_annotation semantics or HF-card counts disagree."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_cache_dir() -> Path:
    return _repo_root() / "data" / "raw" / "toxicchat0124"


def _parse_openai_moderation(raw: Any) -> list[tuple[str, float]]:
    if isinstance(raw, list):
        return [(str(k), float(v)) for k, v in raw]
    if not isinstance(raw, str):
        raise TypeError(f"openai_moderation has unexpected type {type(raw)}")
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse openai_moderation: {raw[:120]!r}") from exc
    return [(str(k), float(v)) for k, v in parsed]


def openai_max_score(raw: Any) -> float:
    """Maximum category score from the in-data OpenAI moderation payload."""
    pairs = _parse_openai_moderation(raw)
    if not pairs:
        return 0.0
    return max(score for _, score in pairs)


def _split_to_frame(split: Any) -> pd.DataFrame:
    return pd.DataFrame({c: split[c] for c in split.column_names})


def load_toxicchat(
    *,
    cache_dir: Path | None = None,
    force_download: bool = False,
    strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, LoadReport]:
    """Download toxicchat0124, verify, and return (all_rows, human_population, report).

    ``all_rows`` is train+test pooled. ``human_population`` is the analysis
    population: rows with ``human_annotation == True``.

    If semantics are wrong or counts disagree with the HF card, raises
    ``ToxicChatLoadError`` when ``strict=True`` (default).
    """
    cache = cache_dir or default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    parquet_path = cache / "pooled.parquet"
    report_path = cache / "load_report.json"

    if parquet_path.exists() and report_path.exists() and not force_download:
        all_rows = pd.read_parquet(parquet_path)
        report = LoadReport(**json.loads(report_path.read_text()))
    else:
        from datasets import load_dataset

        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124")
        train = _split_to_frame(ds["train"])
        test = _split_to_frame(ds["test"])
        train["split"] = "train"
        test["split"] = "test"
        all_rows = pd.concat([train, test], ignore_index=True)
        report = verify_toxicchat(all_rows, n_train=len(train), n_test=len(test))
        all_rows.to_parquet(parquet_path, index=False)
        report_path.write_text(report.to_json() + "\n")

    if strict and (not report.semantics_ok or not report.counts_match_hf_card):
        raise ToxicChatLoadError(
            "ToxicChat load check failed.\n" + report.to_json()
        )

    human = all_rows.loc[all_rows["human_annotation"].astype(bool)].copy()
    human = human.reset_index(drop=True)
    return all_rows, human, report


def verify_toxicchat(
    all_rows: pd.DataFrame,
    *,
    n_train: int | None = None,
    n_test: int | None = None,
) -> LoadReport:
    """Check column presence, human_annotation semantics, and HF-card counts."""
    notes: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in all_rows.columns]
    if missing:
        raise ToxicChatLoadError(f"Missing columns: {missing}")

    ha = all_rows["human_annotation"]
    semantics_ok = True
    if ha.dtype != bool and set(ha.dropna().unique()) - {True, False, 0, 1}:
        semantics_ok = False
        notes.append(
            f"human_annotation has unexpected values: {sorted(map(str, ha.unique()))}"
        )
    if ha.isna().any():
        semantics_ok = False
        notes.append("human_annotation contains nulls")

    # Semantics: True marks rows with human labels (ToxicChat paper + HF card).
    n_total = len(all_rows)
    n_human = int(ha.astype(bool).sum())
    n_auto = n_total - n_human
    human_frac = n_human / n_total if n_total else float("nan")

    tox_all = float(all_rows["toxicity"].astype(float).mean())
    jb_all = float(all_rows["jailbreaking"].astype(float).mean())
    human_mask = ha.astype(bool)
    tox_h = float(all_rows.loc[human_mask, "toxicity"].astype(float).mean())
    jb_h = float(all_rows.loc[human_mask, "jailbreaking"].astype(float).mean())

    counts_match = (
        n_total == EXPECTED_N_TOTAL and n_human == EXPECTED_N_HUMAN
    )
    if n_train is not None and n_train != EXPECTED_TRAIN:
        counts_match = False
        notes.append(f"train size {n_train} != HF card {EXPECTED_TRAIN}")
    if n_test is not None and n_test != EXPECTED_TEST:
        counts_match = False
        notes.append(f"test size {n_test} != HF card {EXPECTED_TEST}")
    if not counts_match:
        notes.append(
            f"pooled counts n_total={n_total}, n_human={n_human}; "
            f"HF card expects {EXPECTED_N_TOTAL}, {EXPECTED_N_HUMAN}"
        )

    # Paper first-stage ~61% unlabeled vs released 0124 fraction.
    released_unlabeled = 1.0 - human_frac
    notes.append(
        "Paper (arXiv 2310.17389) describes a first-stage process where about "
        f"{PAPER_FIRST_STAGE_UNLABELED_FRAC:.0%} of prompts were never shown to "
        "humans (Perspective below ~0.037). The released toxicchat0124 set has "
        f"{released_unlabeled:.1%} with human_annotation=False "
        f"({n_auto}/{n_total}), matching the HF card "
        f"({EXPECTED_N_HUMAN} human annotations). "
        "We treat HF-card counts as authoritative for this subset and keep "
        "population = human_annotation==True."
    )
    notes.append(
        f"Paper overall rates (context): tox {PAPER_TOX_RATE:.2%}, "
        f"jb {PAPER_JB_RATE:.2%}. Observed all-rows: tox {tox_all:.2%}, "
        f"jb {jb_all:.2%} (HF card: 7.33%, 2.01%). "
        f"Human-annotated subset: tox {tox_h:.2%}, jb {jb_h:.2%}."
    )
    notes.append(
        "Judge score: openai_moderation summarized as max category score "
        "(documented in lee.judges.openai_moderation)."
    )

    if n_train is None and "split" in all_rows.columns:
        n_train = int((all_rows["split"] == "train").sum())
        n_test = int((all_rows["split"] == "test").sum())

    return LoadReport(
        n_train=int(n_train or -1),
        n_test=int(n_test or -1),
        n_total=n_total,
        n_human=n_human,
        n_auto=n_auto,
        human_frac=human_frac,
        tox_rate_all=tox_all,
        jb_rate_all=jb_all,
        tox_rate_human=tox_h,
        jb_rate_human=jb_h,
        paper_tox_rate=PAPER_TOX_RATE,
        paper_jb_rate=PAPER_JB_RATE,
        paper_first_stage_unlabeled_frac=PAPER_FIRST_STAGE_UNLABELED_FRAC,
        hf_card_n_total=EXPECTED_N_TOTAL,
        hf_card_n_human=EXPECTED_N_HUMAN,
        semantics_ok=semantics_ok,
        counts_match_hf_card=counts_match,
        notes=tuple(notes),
    )
