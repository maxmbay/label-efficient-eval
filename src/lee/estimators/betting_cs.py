"""Hedged betting confidence sequences for a bounded mean.

Implements the predictable plug-in bets of Waudby-Smith and Ramdas
(JRSS-B 2024, Theorem 3), evaluated on a grid of candidate means.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfidenceSequenceState:
    """Running CS after t observations."""

    t: int
    lower: float
    upper: float
    alpha: float


def affine_to_unit(z: np.ndarray, *, z_min: float, z_max: float) -> np.ndarray:
    """Map values with known bounds into [0, 1]."""
    if z_max <= z_min:
        raise ValueError("z_max must exceed z_min")
    z = np.asarray(z, dtype=float).ravel()
    return (z - z_min) / (z_max - z_min)


def affine_from_unit(u: float, *, z_min: float, z_max: float) -> float:
    return float(z_min + u * (z_max - z_min))


def ppi_rectifier_bounds(*, lam: float = 1.0) -> tuple[float, float]:
    """Bounds on Y - lam*(f - f_bar) for Y,f in [0,1] and f_bar in [0,1].

    Worst case: Y - lam*f + lam*f_bar ranges in
    [0 - lam*1 + lam*0, 1 - lam*0 + lam*1] = [-lam, 1 + lam].
    """
    return (-float(lam), 1.0 + float(lam))


class HedgedBettingCS:
    """Running intersection of hedged betting CS intervals on [0, 1] data."""

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        grid: int = 2001,
        theta: float = 0.5,
        c: float = 0.5,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if grid < 3:
            raise ValueError("grid must be >= 3")
        self.alpha = float(alpha)
        self.theta = float(theta)
        self.c = float(c)
        self.m_grid = np.linspace(0.0, 1.0, grid)
        # Avoid exact 0/1 for betting denominators.
        eps = 1e-9
        self.m_grid = np.clip(self.m_grid, eps, 1.0 - eps)
        self._wealth_pos = np.ones(self.m_grid.size, dtype=float)
        self._wealth_neg = np.ones(self.m_grid.size, dtype=float)
        self._sum_x = 0.0
        self._sum_var = 0.0
        self._t = 0
        self._lower = 0.0
        self._upper = 1.0
        self._alive = np.ones(self.m_grid.size, dtype=bool)

    @property
    def t(self) -> int:
        return self._t

    @property
    def lower(self) -> float:
        return self._lower

    @property
    def upper(self) -> float:
        return self._upper

    def state(self) -> ConfidenceSequenceState:
        return ConfidenceSequenceState(
            t=self._t,
            lower=self._lower,
            upper=self._upper,
            alpha=self.alpha,
        )

    def update(self, x: float) -> ConfidenceSequenceState:
        """Incorporate one observation in [0, 1]; return the running CS."""
        x = float(x)
        if not (0.0 <= x <= 1.0):
            raise ValueError(f"observation must be in [0, 1], got {x}")

        t_prev = self._t
        # Predictable mean / variance estimates (Waudby-Smith and Ramdas).
        mu_hat = (0.5 + self._sum_x) / (t_prev + 1)
        sig2 = (0.25 + self._sum_var) / (t_prev + 1)
        sig2 = max(sig2, 1e-6)
        i = t_prev + 1
        lam_dot = np.sqrt(
            (2.0 * np.log(2.0 / self.alpha))
            / (sig2 * i * np.log(1.0 + i))
        )

        m = self.m_grid
        alive = self._alive
        lam_pos = np.minimum(lam_dot, self.c / m)
        lam_neg = np.minimum(lam_dot, self.c / (1.0 - m))

        # Only update candidates still inside the running intersection.
        idx = np.where(alive)[0]
        if idx.size:
            self._wealth_pos[idx] *= 1.0 + lam_pos[idx] * (x - m[idx])
            self._wealth_neg[idx] *= 1.0 - lam_neg[idx] * (x - m[idx])
            # Numerical floor.
            self._wealth_pos[idx] = np.maximum(self._wealth_pos[idx], 0.0)
            self._wealth_neg[idx] = np.maximum(self._wealth_neg[idx], 0.0)

        wealth = np.maximum(
            self.theta * self._wealth_pos,
            (1.0 - self.theta) * self._wealth_neg,
        )
        # Reject when wealth >= 1/alpha.
        rejected = wealth >= (1.0 / self.alpha)
        self._alive &= ~rejected

        # Running intersection: widen to adjacent grid step for conservatism.
        if not np.any(self._alive):
            # Degenerate; keep last interval.
            pass
        else:
            alive_m = m[self._alive]
            step = float(m[1] - m[0])
            lo = float(alive_m.min()) - step
            hi = float(alive_m.max()) + step
            self._lower = max(self._lower, max(0.0, lo))
            self._upper = min(self._upper, min(1.0, hi))
            # Enforce running intersection on the alive mask.
            self._alive &= (m >= self._lower - 1e-15) & (m <= self._upper + 1e-15)

        self._t = i
        self._sum_x += x
        # Use sequential centered squares with updated mu for next step's var.
        mu_next = (0.5 + self._sum_x) / (self._t + 1)
        self._sum_var += (x - mu_next) ** 2
        return self.state()

    def update_many(self, xs: np.ndarray) -> list[ConfidenceSequenceState]:
        return [self.update(float(x)) for x in np.asarray(xs, dtype=float).ravel()]


def first_clear_above(
    states: list[ConfidenceSequenceState],
    *,
    threshold: float,
) -> int | None:
    """First t where CS.lower > threshold; None if never."""
    for s in states:
        if s.lower > threshold:
            return s.t
    return None


def first_excludes(
    states: list[ConfidenceSequenceState],
    *,
    threshold: float,
) -> int | None:
    """First t where the CS excludes the threshold."""
    for s in states:
        if s.upper < threshold or s.lower > threshold:
            return s.t
    return None
