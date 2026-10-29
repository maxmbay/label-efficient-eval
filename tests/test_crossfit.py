"""Cross-fit Platt calibration tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lee.judges.crossfit import crossfit_platt


def test_crossfit_platt_uses_both_splits() -> None:
    rng = np.random.default_rng(0)
    n = 200
    split = np.array(["train"] * 100 + ["test"] * 100)
    # Score correlates with label
    y = rng.binomial(1, 0.3, size=n).astype(float)
    raw = np.clip(0.2 + 0.6 * y + 0.1 * rng.random(n), 0, 1)
    frame = pd.DataFrame({"split": split, "toxicity": y})
    cal = crossfit_platt(frame, raw, label_col="toxicity")
    assert cal.shape == (n,)
    assert np.all((cal >= 0) & (cal <= 1))
    # Should be closer to labels than a constant mid score on average.
    assert float(np.mean(np.abs(cal - y))) < 0.45


def test_open_it_prompt_contains_definition() -> None:
    from lee.judges.definitions import TOXICITY_DEFINITION
    from lee.judges.open_it import build_open_it_prompt

    p = build_open_it_prompt("hello", estimand="toxicity")
    assert "toxic" in p.lower()
    assert TOXICITY_DEFINITION[:40] in p
