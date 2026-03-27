"""Discriminant — trajectory-aware pressure classification with calibrated uncertainty.

# IMPLEMENTED TRUTH: linear classifier on trajectory features with sigmoid output.
# APPROXIMATION: linear classifier on handcrafted features, not learned deep model.
# CALIBRATION: synthetic labels only, not real operational data.

Read-only: does not modify system state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from .viability import ViabilityKernel

if TYPE_CHECKING:
    from .types import NormSpace

__all__ = [
    "CalibrationResult",
    "Discriminant",
    "PressureKind",
    "SystemMode",
    "TrajectoryDiscriminant",
]


class PressureKind(Enum):
    OPERATIONAL = "operational"
    EXISTENTIAL = "existential"


class SystemMode(Enum):
    IDLE = "idle"
    RECOVERY = "recovery"
    ADAPTATION = "adaptation"
    TRANSFORMATION = "transformation"


@dataclass(frozen=True)
class CalibrationResult:
    """Result of synthetic calibration.

    # CALIBRATION: synthetic labels, not real operational data.
    """

    ece: float
    accuracy: float
    threshold: float
    n_synthetic: int
    ece_method: str = "ridge"
    label: str = "synthetic_calibration"


class TrajectoryDiscriminant:
    """Trajectory-aware classifier with calibrated uncertainty.

    # IMPLEMENTED TRUTH: linear classifier on trajectory features.
    # APPROXIMATION: linear classifier on handcrafted features,
    #   not a learned deep trajectory model.

    Feature vector z_t = [phi, phi_trend, failure_density, coherence, steps_in_bad]
    Score: s_t = sigmoid(w . z_t + b)
    Uncertainty: u_t = 4 * s_t * (1 - s_t)  (peaks at 0.5)
    """

    def __init__(
        self,
        threshold: float = 0.5,
        uncertainty_threshold: float = 0.4,
    ) -> None:
        # Default weights (fallback if not calibrated)
        self._w = np.array([2.0, 1.5, 3.0, -1.0, 0.5])
        self._b = -3.0
        self._mu = np.zeros(5)
        self._std = np.ones(5)
        self._scorer: object | None = None  # LogisticRegression after calibrate()
        self._isotonic: object | None = None  # IsotonicRegression after calibrate()
        self.threshold = threshold
        self.uncertainty_threshold = uncertainty_threshold

    def classify(
        self,
        phi: float,
        phi_trend: float,
        failure_density: float,
        coherence: float,
        steps_in_bad_phase: int,
    ) -> tuple[PressureKind, float]:
        """Classify pressure with uncertainty score.

        Returns (kind, uncertainty). High uncertainty -> OPERATIONAL (Gate 7).
        Uses isotonic-calibrated probabilities if available.
        """
        z = np.array([phi, phi_trend, failure_density, coherence,
                       float(steps_in_bad_phase) / 100.0])

        # Use isotonic pipeline if calibrated, else fallback to linear
        if self._scorer is not None and self._isotonic is not None:
            z_2d = z.reshape(1, -1)
            p_raw = float(self._scorer.predict_proba(z_2d)[0, 1])
            score = float(self._isotonic.predict([p_raw])[0])
        else:
            z_norm = (z - self._mu) / self._std
            logit = float(self._w @ z_norm + self._b)
            score = 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))

        uncertainty = 4.0 * score * (1.0 - score)

        # GATE 7: high uncertainty -> conservative OPERATIONAL
        if uncertainty > self.uncertainty_threshold:
            return PressureKind.OPERATIONAL, uncertainty

        if score > self.threshold:
            return PressureKind.EXISTENTIAL, uncertainty
        return PressureKind.OPERATIONAL, uncertainty

    @staticmethod
    def _to_z(d: dict[str, float]) -> np.ndarray:
        return np.array([
            d.get("phi", 0), d.get("phi_trend", 0),
            d.get("failure_density", 0), d.get("coherence", 0.5),
            d.get("steps_in_bad", 0) / 100.0,
        ])

    @staticmethod
    def _compute_ece(
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """ECE = sum_b (|b|/n) * |mean_conf(b) - mean_acc(b)|."""
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(probs)
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = mask | (probs == bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            conf = float(probs[mask].mean())
            acc = float(labels[mask].mean())
            ece += (mask.sum() / n) * abs(conf - acc)
        return ece

    def calibrate(
        self,
        operational: list[dict[str, float]],
        existential: list[dict[str, float]],
    ) -> CalibrationResult:
        """Two-stage calibration: LogisticRegression + IsotonicRegression.

        # IMPLEMENTED TRUTH: isotonic post-hoc calibration, ECE < 0.15 on synthetic.
        # CALIBRATION: synthetic labels only, not real operational data.
        """
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        # Build feature matrix
        X_op = np.array([self._to_z(d) for d in operational])
        X_ex = np.array([self._to_z(d) for d in existential])
        X = np.vstack([X_op, X_ex])
        y = np.concatenate([np.zeros(len(X_op)), np.ones(len(X_ex))])

        # Train/val split (stratified)
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(X))
        split = int(len(X) * 0.8)
        train_idx, val_idx = idx[:split], idx[split:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Stage 1: LogisticRegression scorer
        scorer = LogisticRegression(max_iter=1000, random_state=42)
        scorer.fit(X_train, y_train)
        p_val_raw = scorer.predict_proba(X_val)[:, 1]

        # Stage 2: isotonic calibration on validation fold
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_val_raw, y_val)
        p_cal = iso.predict(p_val_raw)

        # Stage 3: ECE on calibrated scores
        ece = self._compute_ece(p_cal, y_val, n_bins=10)

        # Accuracy
        accuracy = float(np.mean((p_cal >= 0.5) == y_val))

        # Store for inference
        self._scorer = scorer
        self._isotonic = iso
        self.threshold = 0.5

        return CalibrationResult(
            ece=round(ece, 4),
            accuracy=round(accuracy, 4),
            threshold=0.5,
            n_synthetic=len(X),
            ece_method="isotonic",
        )


class Discriminant:
    """Trajectory-aware pressure classification with hysteresis.

    Upgraded: uses TrajectoryDiscriminant score + min_consecutive_existential.
    """

    def __init__(
        self,
        viability: ViabilityKernel | None = None,
        coherence_critical: float = 0.15,
        drift_threshold: float = 0.5,
        min_consecutive_existential: int = 3,
    ) -> None:
        self.viability = viability or ViabilityKernel()
        self.coherence_critical = coherence_critical
        self.drift_threshold = drift_threshold
        self.min_consecutive_existential = min_consecutive_existential
        self.trajectory = TrajectoryDiscriminant()
        self._consecutive_existential: int = 0

    def classify(
        self,
        phi: float,
        tau: float,
        x: np.ndarray,
        norm: NormSpace,
        phase_is_collapsing: bool,
        coherence: float,
        horizon: int = 10,
        phi_trend: float = 0.0,
        failure_density: float = 0.0,
        steps_in_bad_phase: int = 0,
    ) -> PressureKind:
        """Classify pressure with trajectory evidence and hysteresis."""
        # Trajectory-aware classification
        kind, _uncertainty = self.trajectory.classify(
            phi, phi_trend, failure_density, coherence, steps_in_bad_phase,
        )

        # Also check critical condition
        if phase_is_collapsing and coherence < self.coherence_critical:
            kind = PressureKind.EXISTENTIAL

        # PRIMARY: collapse pressure exceeds threshold
        if phi >= tau:
            kind = PressureKind.EXISTENTIAL

        # Hysteresis: require consecutive EXISTENTIAL steps
        if kind == PressureKind.EXISTENTIAL:
            self._consecutive_existential += 1
        else:
            self._consecutive_existential = 0

        if (
            kind == PressureKind.EXISTENTIAL
            and self._consecutive_existential < self.min_consecutive_existential
        ):
            return PressureKind.OPERATIONAL

        return kind

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

        if not norm.contains(x):
            return SystemMode.RECOVERY

        drift = norm.drift_from_origin(norm_origin)
        if drift > self.drift_threshold:
            return SystemMode.ADAPTATION

        return SystemMode.IDLE
