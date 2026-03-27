"""Discriminant — trajectory-aware pressure classification.

D_t classifies pressure via trajectory, not single state:
  PRIMARY:   Phi >= tau                              -> EXISTENTIAL
  GEOMETRIC: x not in capture_basin(S, W)            -> EXISTENTIAL
  CRITICAL:  phase==COLLAPSING AND coherence < 0.15  -> EXISTENTIAL
  ELSE:                                               -> OPERATIONAL

SystemMode from PressureKind:
  EXISTENTIAL             -> TRANSFORMATION
  x not in S, OPERATIONAL -> RECOVERY
  drift detected          -> ADAPTATION
  ELSE                    -> IDLE

Read-only: does not modify system state.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from .viability import ViabilityKernel

if TYPE_CHECKING:
    import numpy as np

    from .types import NormSpace

__all__ = ["Discriminant", "PressureKind", "SystemMode"]


class PressureKind(Enum):
    OPERATIONAL = "operational"
    EXISTENTIAL = "existential"


class SystemMode(Enum):
    IDLE = "idle"
    RECOVERY = "recovery"
    ADAPTATION = "adaptation"
    TRANSFORMATION = "transformation"


class Discriminant:
    """Trajectory-aware pressure classification."""

    def __init__(
        self,
        viability: ViabilityKernel | None = None,
        coherence_critical: float = 0.15,
        drift_threshold: float = 0.5,
    ) -> None:
        self.viability = viability or ViabilityKernel()
        self.coherence_critical = coherence_critical
        self.drift_threshold = drift_threshold

    def classify(
        self,
        phi: float,
        tau: float,
        x: np.ndarray,
        norm: NormSpace,
        phase_is_collapsing: bool,
        coherence: float,
        horizon: int = 10,
    ) -> PressureKind:
        """Classify current pressure as OPERATIONAL or EXISTENTIAL."""
        # PRIMARY: collapse pressure exceeds threshold (trajectory-aware)
        if phi >= tau:
            return PressureKind.EXISTENTIAL

        # CRITICAL: collapsing with very low coherence
        if phase_is_collapsing and coherence < self.coherence_critical:
            return PressureKind.EXISTENTIAL

        # GEOMETRIC alone is NOT existential — it's a recovery signal.
        # Only existential when combined with sustained collapse pressure.
        if (
            not self.viability.in_capture_basin(x, norm, horizon)
            and phase_is_collapsing
            and phi > tau * 0.5
        ):
            return PressureKind.EXISTENTIAL

        return PressureKind.OPERATIONAL

    def mode_from_state(
        self,
        pressure: PressureKind,
        x: np.ndarray,
        norm: NormSpace,
        norm_origin: NormSpace,
    ) -> SystemMode:
        """Determine system mode from pressure and state."""
        if pressure == PressureKind.EXISTENTIAL:
            return SystemMode.TRANSFORMATION

        # x outside norm but operational -> recovery
        if not norm.contains(x):
            return SystemMode.RECOVERY

        # Drift detected -> adaptation
        drift = norm.drift_from_origin(norm_origin)
        if drift > self.drift_threshold:
            return SystemMode.ADAPTATION

        return SystemMode.IDLE
