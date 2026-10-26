"""In-data OpenAI moderation scores (no API spend).

Summary rule (locked for M1): for each prompt, take the **maximum category
score** across the OpenAI moderation categories stored in
``openai_moderation``. This is a soft unsafe score in [0, 1]. Perspective API
is never used (it defined ToxicChat's selection).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lee.data.toxicchat import openai_max_score


class OpenAIModerationMaxJudge:
    """Max category score from the ToxicChat ``openai_moderation`` column."""

    name = "openai_moderation_max"

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        if "openai_moderation" not in frame.columns:
            raise KeyError("frame missing openai_moderation")
        return np.asarray(
            [openai_max_score(v) for v in frame["openai_moderation"]],
            dtype=float,
        )
