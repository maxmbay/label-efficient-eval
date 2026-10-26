"""Shared estimator result types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntervalEstimate:
    """Point estimate with a two-sided interval."""

    estimate: float
    lower: float
    upper: float
    alpha: float
    method: str
    lam: float | None = None

    @property
    def width(self) -> float:
        return float(self.upper - self.lower)
