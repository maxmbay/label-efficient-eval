"""M2 headline must not invent savings when coverage fails."""

from __future__ import annotations

import pandas as pd

from lee.sim.headline_m2 import compute_m2_headline


def test_headline_reports_failure_when_overcover() -> None:
    metrics = pd.DataFrame(
        [
            {
                "estimator": "ppi_pp",
                "judge": "gpt4_pair",
                "estimand": "winrate:x_vs_y",
                "n": 50,
                "R": 1000,
                "alpha": 0.05,
                "coverage": 0.989,
                "coverage_se": 0.003,
                "mean_width": 0.25,
                "bias": 0.0,
                "rmse": 0.0,
                "mean_lam": 0.2,
            },
            {
                "estimator": "classical_clustered",
                "judge": "gpt4_pair",
                "estimand": "winrate:x_vs_y",
                "n": 50,
                "R": 1000,
                "alpha": 0.05,
                "coverage": 0.99,
                "coverage_se": 0.003,
                "mean_width": 0.26,
                "bias": 0.0,
                "rmse": 0.0,
                "mean_lam": 0.0,
            },
        ]
    )
    h = compute_m2_headline(metrics, n=50)
    assert h.labels_saved is None
    assert "Do not report a labels-saved" in h.message
