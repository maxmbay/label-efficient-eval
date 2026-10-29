"""Stub judges for the quality ladder (Modal / HF runs come later)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Deliberately weak keyword list for the bottom of the ladder.
_WEAK_TOXIC_PATTERNS = [
    r"\bkill\b",
    r"\bhate\b",
    r"\bstupid\b",
    r"\bidiot\b",
    r"\bfuck\b",
    r"\bshit\b",
    r"\basshole\b",
    r"\bnazi\b",
    r"\bracist\b",
    r"\bsuicide\b",
    r"\bbomb\b",
    r"\battack\b",
]
_WEAK_RE = re.compile("|".join(_WEAK_TOXIC_PATTERNS), re.IGNORECASE)


class LlamaGuardStub:
    """Placeholder for Llama Guard 3 on Modal (gated HF weights).

    Returns NaNs until Modal / HF credentials are configured. Simulation
    skips judges that raise ``NotImplementedError``.
    """

    name = "llama_guard3_stub"

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError(
            "Llama Guard 3 requires Modal + Hugging Face license accept. "
            "Not run in Milestone 1 smoke; wire in a later PR."
        )


class OpenInstructStub:
    """Placeholder for a zero-shot instruction-tuned open judge."""

    name = "open_it_stub"

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError(
            "Open instruction-tuned judge not run in Milestone 1 smoke. "
            "Prompt with ToxicChat toxicity/jailbreak definitions when wired."
        )


class WeakKeywordJudge:
    """Keyword hit-rate proxy: 1.0 if any weak toxic keyword matches, else 0.0."""

    name = "weak_keyword"

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        texts = frame["user_input"].astype(str)
        return np.asarray(
            [1.0 if _WEAK_RE.search(t) else 0.0 for t in texts],
            dtype=float,
        )
