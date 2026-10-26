"""Shared estimator result types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntervalEstimate:
    """Point estimate with a two-sided confidence interval."""

    estimate: float
    lower: float
    upper: float
    alpha: float
    method: str
    lam: float | None = None

    @property
    def width(self) -> float:
        return float(self.upper - self.lower)

    def covers(self, truth: float) -> bool:
        return bool(self.lower <= truth <= self.upper)
