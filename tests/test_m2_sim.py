"""Milestone 2 simulation smoke (uses on-disk / network MT-Bench cache)."""

from __future__ import annotations

import numpy as np
import pytest

from lee.data.mt_bench import PRIMARY_PAIR, build_pair_population, load_mt_bench
from lee.sim.m2 import M2Config, run_m2_grid, subset_seed


def test_subset_seed_m2_offset() -> None:
    assert subset_seed(50, 0) == 20261026 + 2_000_003 * 50


@pytest.mark.integration
def test_m2_grid_smoke() -> None:
    human, gpt4, _c, _r = load_mt_bench(strict=True)
    pop = build_pair_population(
        human, gpt4, model_lo=PRIMARY_PAIR[0], model_hi=PRIMARY_PAIR[1]
    )
    assert len(pop) == 105
    assert set(pop.columns) >= {"y", "f", "question_id"}
    cfg = M2Config(R=3, n_grid=(30, 50), model_lo=PRIMARY_PAIR[0], model_hi=PRIMARY_PAIR[1])
    metrics = run_m2_grid(population=pop, config=cfg)
    assert metrics
    methods = {m.estimator for m in metrics}
    assert "classical_clustered" in methods
    assert "ppi_pp" in methods
    assert "judge_only" in methods
    # n=200 must be skipped for this pair
    cfg_skip = M2Config(R=1, n_grid=(200,), model_lo=PRIMARY_PAIR[0], model_hi=PRIMARY_PAIR[1])
    assert run_m2_grid(population=pop, config=cfg_skip) == []
    # truth finite
    assert 0.0 < float(np.mean(pop["y"])) < 1.0
