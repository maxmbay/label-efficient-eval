"""Bradley-Terry unit tests on synthetic tournaments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lee.estimators.bradley_terry import fit_bradley_terry, fit_bradley_terry_mle


def test_bt_mle_recovers_ordering() -> None:
    # A beats B always; B beats C always; A beats C always
    y = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    a = np.array([0, 0, 1, 0, 0, 1])
    b = np.array([1, 2, 2, 1, 2, 2])
    theta = fit_bradley_terry_mle(y, a, b, n_models=3)
    assert theta[0] > theta[1] > theta[2]


def test_bt_frame_bootstrap() -> None:
    rows = []
    for q in range(20):
        rows.append(
            {
                "question_id": q,
                "model_lo": "a",
                "model_hi": "b",
                "majority_winner": "model_a" if q % 3 else "model_b",
            }
        )
        rows.append(
            {
                "question_id": q,
                "model_lo": "b",
                "model_hi": "c",
                "majority_winner": "model_a",
            }
        )
    frame = pd.DataFrame(rows)
    result = fit_bradley_terry(frame, n_bootstrap=50, seed=0)
    assert set(result.strengths) == {"a", "b", "c"}
    assert result.n_comparisons == 40
    assert all(np.isfinite(list(result.se.values())))
