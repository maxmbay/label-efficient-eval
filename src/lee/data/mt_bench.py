"""Load and verify lmsys/mt_bench_human_judgments.

Milestone 2 population helpers: human votes, GPT-4 pairwise votes, category
labels from FastChat's MT-Bench question file. Raw data is cached under
data/raw/ and data/cache/ (gitignored); never committed.
"""

from __future__ import annotations

import json
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

DATASET_ID = "lmsys/mt_bench_human_judgments"

# Verified 2026-10-26 against the Hugging Face dataset (5755 rows total).
EXPECTED_N_HUMAN = 3355
EXPECTED_N_GPT4 = 2400
EXPECTED_N_QUESTIONS = 80
EXPECTED_QUESTION_ID_MIN = 81
EXPECTED_QUESTION_ID_MAX = 160

EXPECTED_MODELS = frozenset(
    {
        "alpaca-13b",
        "claude-v1",
        "gpt-3.5-turbo",
        "gpt-4",
        "llama-13b",
        "vicuna-13b-v1.2",
    }
)

REQUIRED_COLUMNS = (
    "question_id",
    "model_a",
    "model_b",
    "winner",
    "judge",
    "conversation_a",
    "conversation_b",
    "turn",
)

HUMAN_WINNERS = frozenset({"model_a", "model_b", "tie"})
# GPT-4 split includes "tie (inconsistent)" when the two presentation orders
# disagree (Zheng et al., MT-Bench / Chatbot Arena).
GPT4_WINNERS = frozenset({"model_a", "model_b", "tie", "tie (inconsistent)"})

QUESTION_JSONL_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)


@dataclass(frozen=True)
class MTBenchLoadReport:
    """Verification summary written beside the cached tables."""

    n_human: int
    n_gpt4: int
    n_questions_human: int
    n_questions_gpt4: int
    n_human_judges: int
    models: tuple[str, ...]
    human_winner_counts: dict[str, int]
    gpt4_winner_counts: dict[str, int]
    human_tie_rate: float
    gpt4_tie_rate: float
    gpt4_inconsistent_tie_rate: float
    n_units_human: int
    n_units_multi_human: int
    inter_human_agree_3way: float
    inter_human_agree_nontie: float
    n_decisive_majority_units: int
    n_majority_tie_units: int
    n_all_tie_units: int
    categories: tuple[str, ...]
    columns_ok: bool
    counts_match_expected: bool
    semantics_ok: bool
    notes: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class MTBenchLoadError(RuntimeError):
    """Raised when MT-Bench load checks fail under strict mode."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_cache_dir() -> Path:
    return _repo_root() / "data" / "raw" / "mt_bench_human_judgments"


def default_question_cache_path() -> Path:
    return _repo_root() / "data" / "cache" / "mt_bench" / "question.jsonl"


def _split_to_frame(split: Any) -> pd.DataFrame:
    return pd.DataFrame({c: split[c] for c in split.column_names})


def canonicalize_pair(
    model_a: str,
    model_b: str,
    winner: str,
) -> tuple[str, str, str, bool]:
    """Return (model_lo, model_hi, winner_canon, swapped).

    Winner is remapped into {model_a, model_b, tie, tie (inconsistent)} relative
    to the canonical (lexicographic) order. ``swapped`` is True when the
    original presentation had model_a > model_b lexicographically.
    """
    if model_a == model_b:
        raise ValueError(f"model_a and model_b are identical: {model_a!r}")
    if model_a < model_b:
        return model_a, model_b, winner, False
    if winner == "model_a":
        w = "model_b"
    elif winner == "model_b":
        w = "model_a"
    else:
        w = winner
    return model_b, model_a, w, True


def add_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add model_lo, model_hi, winner_canon, swapped columns."""
    out = df.copy()
    triples = [
        canonicalize_pair(a, b, w)
        for a, b, w in zip(out["model_a"], out["model_b"], out["winner"])
    ]
    out["model_lo"] = [t[0] for t in triples]
    out["model_hi"] = [t[1] for t in triples]
    out["winner_canon"] = [t[2] for t in triples]
    out["swapped"] = [t[3] for t in triples]
    return out


def unit_key_columns() -> tuple[str, ...]:
    return ("question_id", "turn", "model_lo", "model_hi")


def majority_label(labels: Iterable[str]) -> str:
    """Majority among non-tie labels; else 'tie' / 'majority_tie'.

    Returns one of: model_a, model_b, tie, majority_tie.
    Input labels should already be in a consistent pair orientation
    (e.g. winner_canon).
    """
    labs = list(labels)
    nontie = [x for x in labs if x not in {"tie", "tie (inconsistent)"}]
    if not nontie:
        return "tie"
    counts = Counter(nontie)
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "majority_tie"
    return top[0][0]


def collapse_human_majority(human: pd.DataFrame) -> pd.DataFrame:
    """One row per canonical unit with majority human label.

    Requires canonical columns from ``add_canonical_columns``.
    """
    keys = list(unit_key_columns())
    rows: list[dict[str, Any]] = []
    for key, grp in human.groupby(keys, sort=False):
        key_dict = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        label = majority_label(grp["winner_canon"])
        rows.append(
            {
                **key_dict,
                "majority_winner": label,
                "n_votes": int(len(grp)),
                "n_ties": int(
                    grp["winner_canon"].isin(["tie", "tie (inconsistent)"]).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def winner_to_score(
    winner: str,
    *,
    perspective: str = "model_a",
    half_credit_ties: bool = False,
) -> float | None:
    """Map a winner label to a score in {0, 0.5, 1} from ``perspective``.

    ``perspective`` is ``model_a`` or ``model_b`` in the same orientation as
    ``winner``. Returns None when the vote is a tie and half-credit is off,
    or when the label is unrecognized.
    """
    if winner in {"tie", "tie (inconsistent)"}:
        return 0.5 if half_credit_ties else None
    if winner == perspective:
        return 1.0
    if winner in {"model_a", "model_b"}:
        return 0.0
    return None


def inter_human_agreement(human: pd.DataFrame) -> dict[str, float | int]:
    """Pairwise percent agreement among human judges on multi-labeled units.

    Uses canonical orientation. Reports three-way agreement and agreement
    restricted to pairs where neither vote is a tie.
    """
    keys = list(unit_key_columns())
    if "winner_canon" not in human.columns:
        raise ValueError("human frame needs winner_canon; call add_canonical_columns")

    agree_3 = total_3 = 0
    agree_nt = total_nt = 0
    n_multi = 0
    for _, grp in human.groupby(keys, sort=False):
        labs = list(grp["winner_canon"])
        if len(labs) < 2:
            continue
        n_multi += 1
        for a, b in combinations(labs, 2):
            total_3 += 1
            agree_3 += int(a == b)
            if a in {"tie", "tie (inconsistent)"} or b in {
                "tie",
                "tie (inconsistent)",
            }:
                continue
            total_nt += 1
            agree_nt += int(a == b)

    return {
        "n_units_multi_human": n_multi,
        "agree_3way_num": agree_3,
        "agree_3way_den": total_3,
        "inter_human_agree_3way": (agree_3 / total_3) if total_3 else float("nan"),
        "agree_nontie_num": agree_nt,
        "agree_nontie_den": total_nt,
        "inter_human_agree_nontie": (agree_nt / total_nt) if total_nt else float("nan"),
    }


def fetch_question_categories(
    *,
    cache_path: Path | None = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """Download FastChat MT-Bench question.jsonl and return id/category frame."""
    path = cache_path or default_question_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if force_download or not path.exists():
        with urllib.request.urlopen(QUESTION_JSONL_URL, timeout=60) as resp:
            path.write_bytes(resp.read())

    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        rows.append(
            {
                "question_id": int(obj["question_id"]),
                "category": str(obj["category"]),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != EXPECTED_N_QUESTIONS:
        raise MTBenchLoadError(
            f"question.jsonl has {len(frame)} rows; expected {EXPECTED_N_QUESTIONS}"
        )
    return frame


def load_mt_bench(
    *,
    cache_dir: Path | None = None,
    force_download: bool = False,
    strict: bool = True,
    fetch_categories: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, MTBenchLoadReport]:
    """Download MT-Bench judgments and return (human, gpt4, categories, report).

    ``human`` and ``gpt4`` include canonical pair columns. ``categories`` is
    None when ``fetch_categories`` is False.
    """
    cache = cache_dir or default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    human_path = cache / "human.parquet"
    gpt4_path = cache / "gpt4_pair.parquet"
    report_path = cache / "load_report.json"

    if (
        human_path.exists()
        and gpt4_path.exists()
        and report_path.exists()
        and not force_download
    ):
        human = pd.read_parquet(human_path)
        gpt4 = pd.read_parquet(gpt4_path)
        report = MTBenchLoadReport(**json.loads(report_path.read_text()))
    else:
        from datasets import load_dataset

        ds = load_dataset(DATASET_ID)
        human = add_canonical_columns(_split_to_frame(ds["human"]))
        gpt4 = add_canonical_columns(_split_to_frame(ds["gpt4_pair"]))
        categories = (
            fetch_question_categories(force_download=force_download)
            if fetch_categories
            else None
        )
        report = verify_mt_bench(human, gpt4, categories=categories)
        # Conversations are large; drop from parquet cache used for stats/sims.
        # Full conversations remain available via force_download reload if needed.
        human_store = human.drop(columns=["conversation_a", "conversation_b"])
        gpt4_store = gpt4.drop(columns=["conversation_a", "conversation_b"])
        human_store.to_parquet(human_path, index=False)
        gpt4_store.to_parquet(gpt4_path, index=False)
        report_path.write_text(report.to_json() + "\n")
        if categories is not None:
            return human_store, gpt4_store, categories, report
        return human_store, gpt4_store, None, report

    categories = (
        fetch_question_categories(force_download=False) if fetch_categories else None
    )
    # Recompute agreement fields if report was loaded from an older schema.
    if strict and (
        not report.semantics_ok
        or not report.columns_ok
        or not report.counts_match_expected
    ):
        raise MTBenchLoadError("MT-Bench load check failed.\n" + report.to_json())

    return human, gpt4, categories, report


# Primary pairwise estimand after load check (analysis_plan.md M2 §4 / §8).
# gpt-3.5-turbo vs gpt-4: GPT-4 on one side, full gpt4_pair overlap, 105
# decisive majority units. Locked before the M2 grid.
PRIMARY_PAIR: tuple[str, str] = ("gpt-3.5-turbo", "gpt-4")


def build_pair_population(
    human: pd.DataFrame,
    gpt4: pd.DataFrame,
    *,
    model_lo: str,
    model_hi: str,
) -> pd.DataFrame:
    """Decisive majority units for one canonical pair, with GPT-4 judge score.

    Columns: unit keys, ``y`` (1 if model_lo wins), ``f`` (GPT-4 score with
    half-credit on ties / inconsistent), ``question_id``, ``turn``.
    """
    if "winner_canon" not in human.columns:
        human = add_canonical_columns(human)
    if "winner_canon" not in gpt4.columns:
        gpt4 = add_canonical_columns(gpt4)

    collapsed = collapse_human_majority(human)
    keys = list(unit_key_columns())
    pair = collapsed[
        (collapsed["model_lo"] == model_lo)
        & (collapsed["model_hi"] == model_hi)
        & (~collapsed["majority_winner"].isin(["tie", "majority_tie"]))
    ].copy()
    if pair.empty:
        raise MTBenchLoadError(f"No decisive units for pair {(model_lo, model_hi)}")

    pair["y"] = (pair["majority_winner"] == "model_a").astype(float)

    g = gpt4[
        (gpt4["model_lo"] == model_lo) & (gpt4["model_hi"] == model_hi)
    ][keys + ["winner_canon"]].copy()
    g["f"] = g["winner_canon"].map(
        lambda w: winner_to_score(w, perspective="model_a", half_credit_ties=True)
    )
    g["f"] = g["f"].astype(float)
    merged = pair.merge(g[keys + ["f"]], on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(pair):
        raise MTBenchLoadError(
            f"GPT-4 judge missing for {len(pair) - len(merged)} / {len(pair)} "
            f"decisive units on {(model_lo, model_hi)}"
        )
    return merged.reset_index(drop=True)


def verify_mt_bench(
    human: pd.DataFrame,
    gpt4: pd.DataFrame,
    *,
    categories: pd.DataFrame | None = None,
) -> MTBenchLoadReport:
    """Check columns, winner semantics, counts, and inter-human agreement."""
    notes: list[str] = []
    columns_ok = True
    semantics_ok = True

    for name, frame in ("human", human), ("gpt4_pair", gpt4):
        missing = [
            c
            for c in REQUIRED_COLUMNS
            if c not in frame.columns and c not in {"conversation_a", "conversation_b"}
        ]
        # conversation_* may be dropped in cache; require core judgment columns.
        core = (
            "question_id",
            "model_a",
            "model_b",
            "winner",
            "judge",
            "turn",
        )
        missing_core = [c for c in core if c not in frame.columns]
        if missing_core:
            columns_ok = False
            notes.append(f"{name} missing columns: {missing_core}")

    human_w = set(human["winner"].astype(str).unique())
    gpt4_w = set(gpt4["winner"].astype(str).unique())
    if not human_w <= HUMAN_WINNERS:
        semantics_ok = False
        notes.append(f"unexpected human winners: {sorted(human_w - HUMAN_WINNERS)}")
    if not gpt4_w <= GPT4_WINNERS:
        semantics_ok = False
        notes.append(f"unexpected gpt4 winners: {sorted(gpt4_w - GPT4_WINNERS)}")

    models = sorted(
        set(human["model_a"]).union(human["model_b"]).union(gpt4["model_a"]).union(
            gpt4["model_b"]
        )
    )
    if set(models) != EXPECTED_MODELS:
        semantics_ok = False
        notes.append(f"models {models} != expected {sorted(EXPECTED_MODELS)}")

    n_human = len(human)
    n_gpt4 = len(gpt4)
    n_q_h = int(human["question_id"].nunique())
    n_q_g = int(gpt4["question_id"].nunique())
    counts_match = (
        n_human == EXPECTED_N_HUMAN
        and n_gpt4 == EXPECTED_N_GPT4
        and n_q_h == EXPECTED_N_QUESTIONS
        and n_q_g == EXPECTED_N_QUESTIONS
    )
    if not counts_match:
        notes.append(
            f"counts human={n_human}, gpt4={n_gpt4}, questions={n_q_h}/{n_q_g}; "
            f"expected {EXPECTED_N_HUMAN}, {EXPECTED_N_GPT4}, {EXPECTED_N_QUESTIONS}"
        )

    human_winner_counts = {
        str(k): int(v) for k, v in human["winner"].value_counts().items()
    }
    gpt4_winner_counts = {
        str(k): int(v) for k, v in gpt4["winner"].value_counts().items()
    }
    human_tie_rate = float((human["winner"] == "tie").mean())
    gpt4_tie_rate = float(
        gpt4["winner"].isin(["tie", "tie (inconsistent)"]).mean()
    )
    gpt4_inconsistent = float((gpt4["winner"] == "tie (inconsistent)").mean())

    if "winner_canon" not in human.columns:
        human = add_canonical_columns(human)
    if "winner_canon" not in gpt4.columns:
        gpt4 = add_canonical_columns(gpt4)

    agree = inter_human_agreement(human)
    collapsed = collapse_human_majority(human)
    n_decisive = int(
        (~collapsed["majority_winner"].isin(["tie", "majority_tie"])).sum()
    )
    n_maj_tie = int((collapsed["majority_winner"] == "majority_tie").sum())
    n_all_tie = int((collapsed["majority_winner"] == "tie").sum())
    n_units = int(len(collapsed))

    cat_names: tuple[str, ...] = ()
    if categories is not None:
        cat_names = tuple(sorted(categories["category"].unique()))
        notes.append(
            f"Categories from FastChat question.jsonl ({QUESTION_JSONL_URL}): "
            + ", ".join(cat_names)
        )
    else:
        notes.append("Categories not loaded (fetch_categories=False).")

    notes.append(
        "Inter-human agreement is the ceiling on any LLM judge. "
        f"Pairwise percent agreement on multi-labeled units: "
        f"{agree['inter_human_agree_3way']:.1%} (3-way including ties), "
        f"{agree['inter_human_agree_nontie']:.1%} (both votes non-tie). "
        "Zheng et al. report human-human agreement near 80% on non-tie "
        "comparisons; GPT-4 reaches a similar level."
    )
    notes.append(
        "GPT-4 split uses winner='tie (inconsistent)' when the two presentation "
        "orders disagree. Treat as a tie class for scoring unless analyzing "
        "position bias separately."
    )
    notes.append(
        f"Majority collapse: {n_units} canonical units; {n_decisive} decisive; "
        f"{n_maj_tie} majority ties; {n_all_tie} all-tie. Primary PPI++ estimand "
        "uses decisive units only (analysis_plan.md Milestone 2 §3)."
    )

    return MTBenchLoadReport(
        n_human=n_human,
        n_gpt4=n_gpt4,
        n_questions_human=n_q_h,
        n_questions_gpt4=n_q_g,
        n_human_judges=int(human["judge"].nunique()),
        models=tuple(models),
        human_winner_counts=human_winner_counts,
        gpt4_winner_counts=gpt4_winner_counts,
        human_tie_rate=human_tie_rate,
        gpt4_tie_rate=gpt4_tie_rate,
        gpt4_inconsistent_tie_rate=gpt4_inconsistent,
        n_units_human=n_units,
        n_units_multi_human=int(agree["n_units_multi_human"]),
        inter_human_agree_3way=float(agree["inter_human_agree_3way"]),
        inter_human_agree_nontie=float(agree["inter_human_agree_nontie"]),
        n_decisive_majority_units=n_decisive,
        n_majority_tie_units=n_maj_tie,
        n_all_tie_units=n_all_tie,
        categories=cat_names,
        columns_ok=columns_ok,
        counts_match_expected=counts_match,
        semantics_ok=semantics_ok,
        notes=tuple(notes),
    )
