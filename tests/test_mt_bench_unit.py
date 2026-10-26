"""Unit tests for MT-Bench helpers (no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from lee.data.mt_bench import (
    add_canonical_columns,
    canonicalize_pair,
    collapse_human_majority,
    inter_human_agreement,
    majority_label,
    winner_to_score,
)


def test_canonicalize_pair_swaps_winner() -> None:
    lo, hi, w, swapped = canonicalize_pair("gpt-4", "alpaca-13b", "model_a")
    assert (lo, hi) == ("alpaca-13b", "gpt-4")
    assert w == "model_b"
    assert swapped is True


def test_canonicalize_pair_preserves_tie() -> None:
    lo, hi, w, swapped = canonicalize_pair("gpt-4", "alpaca-13b", "tie")
    assert (lo, hi) == ("alpaca-13b", "gpt-4")
    assert w == "tie"
    assert swapped is True


def test_majority_label() -> None:
    assert majority_label(["model_a", "model_a", "tie"]) == "model_a"
    assert majority_label(["tie", "tie"]) == "tie"
    assert majority_label(["model_a", "model_b"]) == "majority_tie"


def test_winner_to_score() -> None:
    assert winner_to_score("model_a", perspective="model_a") == 1.0
    assert winner_to_score("model_b", perspective="model_a") == 0.0
    assert winner_to_score("tie", perspective="model_a") is None
    assert winner_to_score("tie", perspective="model_a", half_credit_ties=True) == 0.5
    assert (
        winner_to_score(
            "tie (inconsistent)", perspective="model_a", half_credit_ties=True
        )
        == 0.5
    )


def test_collapse_and_agreement_synthetic() -> None:
    raw = pd.DataFrame(
        {
            "question_id": [81, 81, 81, 82],
            "turn": [1, 1, 1, 1],
            "model_a": ["alpaca-13b", "alpaca-13b", "alpaca-13b", "gpt-4"],
            "model_b": ["gpt-4", "gpt-4", "gpt-4", "llama-13b"],
            "winner": ["model_b", "model_b", "tie", "model_a"],
            "judge": ["expert_0", "expert_1", "expert_2", "expert_0"],
        }
    )
    human = add_canonical_columns(raw)
    collapsed = collapse_human_majority(human)
    assert len(collapsed) == 2
    row81 = collapsed.set_index("question_id").loc[81]
    # alpaca-13b < gpt-4; raw winner model_b means gpt-4; canon keeps model_b.
    assert row81["majority_winner"] == "model_b"

    agree = inter_human_agreement(human)
    assert agree["n_units_multi_human"] == 1
    assert agree["agree_3way_den"] == 3  # C(3,2)
    assert agree["inter_human_agree_3way"] == pytest.approx(1 / 3)
