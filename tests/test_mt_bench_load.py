"""MT-Bench load verification (network / on-disk cache)."""

from __future__ import annotations

import pytest

from lee.data.mt_bench import (
    EXPECTED_N_GPT4,
    EXPECTED_N_HUMAN,
    EXPECTED_N_QUESTIONS,
    EXPECTED_MODELS,
    collapse_human_majority,
    load_mt_bench,
)


@pytest.mark.integration
def test_load_and_verify_mt_bench() -> None:
    human, gpt4, categories, report = load_mt_bench(strict=True)
    assert report.semantics_ok
    assert report.columns_ok
    assert report.counts_match_expected
    assert report.n_human == EXPECTED_N_HUMAN
    assert report.n_gpt4 == EXPECTED_N_GPT4
    assert len(human) == EXPECTED_N_HUMAN
    assert len(gpt4) == EXPECTED_N_GPT4
    assert report.n_questions_human == EXPECTED_N_QUESTIONS
    assert set(report.models) == EXPECTED_MODELS
    # Inter-human non-tie agreement near the ~80% ceiling reported by Zheng et al.
    assert 0.75 <= report.inter_human_agree_nontie <= 0.90
    assert 0.55 <= report.inter_human_agree_3way <= 0.75
    assert report.n_decisive_majority_units > 1000
    assert categories is not None
    assert len(categories) == EXPECTED_N_QUESTIONS
    assert "writing" in set(categories["category"])

    collapsed = collapse_human_majority(human)
    assert len(collapsed) == report.n_units_human
