"""ToxicChat load verification (network / cache)."""

from __future__ import annotations

import pytest

from lee.data.toxicchat import (
    EXPECTED_N_HUMAN,
    EXPECTED_N_TOTAL,
    load_toxicchat,
    openai_max_score,
)


@pytest.mark.integration
def test_load_and_verify_human_annotation() -> None:
    all_rows, human, report = load_toxicchat(strict=True)
    assert report.semantics_ok
    assert report.counts_match_hf_card
    assert report.n_total == EXPECTED_N_TOTAL
    assert report.n_human == EXPECTED_N_HUMAN
    assert len(all_rows) == EXPECTED_N_TOTAL
    assert len(human) == EXPECTED_N_HUMAN
    assert human["human_annotation"].all()
    # Human-annotated subset is enriched vs all-rows rates.
    assert report.tox_rate_human > report.tox_rate_all
    assert report.jb_rate_human > report.jb_rate_all
    # Max score is in [0, 1]
    sample = openai_max_score(human["openai_moderation"].iloc[0])
    assert 0.0 <= sample <= 1.0
