"""Collapse tracker — Phi_t with exponential decay.

Phi_t = sum_{s<=t} 1[irreversible(s)] * lambda^{t-s}

irreversible(s): phase==COLLAPSING AND recovery_attempts >= k_max AND norm NOT restored.
lambda default = 0.95 (half-life ~ 14 steps).

Read-only: does not modify system state.
"""

from __future__ import annotations

__all__ = ["CollapseTracker"]


class CollapseTracker:
    """Tracks cumulative collapse pressure Phi with exponential decay."""

    def __init__(
        self,
        decay: float = 0.95,
        k_max: int = 3,
    ) -> None:
        self.decay = decay
        self.k_max = k_max
        self._phi: float = 0.0
        self._consecutive_failures: int = 0
        self._history: list[float] = []

    @property
    def phi(self) -> float:
        return self._phi

    @property
    def history(self) -> list[float]:
        return list(self._history)

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def phi_trend(self, window: int = 20) -> float:
        """Slope of Phi over recent window. Positive = pressure rising."""
        if len(self._history) < 2:
            return 0.0
        import numpy as np

        recent = self._history[-window:]
        if len(recent) < 2:
            return 0.0
        x = np.arange(len(recent), dtype=float)
        return float(np.polyfit(x, recent, 1)[0])

    def failure_density(self, window: int = 20) -> float:
        """Fraction of irreversible events in recent window."""
        if len(self._history) < 2:
            return 0.0
        recent = self._history[-window:]
        # Count jumps (irreversible events cause phi to jump by ~1.0)
        import numpy as np

        diffs = np.diff(recent)
        n_jumps = int(np.sum(diffs > 0.5))  # decay-adjusted threshold
        return float(n_jumps / max(len(recent), 1))

    def record(
        self,
        phase_is_collapsing: bool,
        recovery_succeeded: bool,
        norm_restored: bool,
    ) -> float:
        """Record one step and return updated Phi.

        An event is irreversible when:
          - phase is COLLAPSING
          - recovery has been attempted k_max times
          - norm is still not restored
        """
        # Decay existing pressure
        self._phi *= self.decay

        # Track consecutive failures
        if phase_is_collapsing and not recovery_succeeded:
            self._consecutive_failures += 1
        elif recovery_succeeded or norm_restored:
            self._consecutive_failures = 0

        # Check irreversibility
        is_irreversible = (
            phase_is_collapsing
            and self._consecutive_failures >= self.k_max
            and not norm_restored
        )

        if is_irreversible:
            self._phi += 1.0
            self._consecutive_failures = 0  # reset after recording

        self._history.append(self._phi)
        return self._phi

    def reset(self) -> None:
        self._phi = 0.0
        self._consecutive_failures = 0
        self._history.clear()
