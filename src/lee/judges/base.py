"""Judge interface: soft scores f(X) in [0, 1] where possible."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class Judge(Protocol):
    name: str

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Return a float score per row (same order as ``frame``)."""
        ...
