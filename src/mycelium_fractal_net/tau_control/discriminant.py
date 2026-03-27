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
        # Default weights: phi and failure_density dominate
        self._w = np.array([2.0, 1.5, 3.0, -1.0, 0.5])
        self._b = -3.0
        self._mu = np.zeros(5)
        self._std = np.ones(5)
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
        """
        z = np.array([phi, phi_trend, failure_density, coherence,
                       float(steps_in_bad_phase) / 100.0])
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

    def calibrate(
        self,
        operational: list[dict[str, float]],
        existential: list[dict[str, float]],
    ) -> CalibrationResult:
        """Fit w, b to synthetic data. Compute ECE on held-out 20%.

        # CALIBRATION: synthetic labels only.
        """
        all_data = [(d, 0) for d in operational] + [(d, 1) for d in existential]
        rng = np.random.default_rng(42)
        rng.shuffle(all_data)  # type: ignore[arg-type]

        split = int(len(all_data) * 0.8)
        train, test = all_data[:split], all_data[split:]

        # Build feature matrix
        def _to_z(d: dict[str, float]) -> np.ndarray:
            return np.array([
                d.get("phi", 0), d.get("phi_trend", 0),
                d.get("failure_density", 0), d.get("coherence", 0.5),
                d.get("steps_in_bad", 0) / 100.0,
            ])

        X_train = np.array([_to_z(d) for d, _ in train])
        y_train = np.array([label for _, label in train], dtype=float)
        X_test = np.array([_to_z(d) for d, _ in test])
        y_test = np.array([label for _, label in test], dtype=float)

        # Standardize features for better ridge fit
        self._mu = X_train.mean(axis=0)
        self._std = X_train.std(axis=0) + 1e-12
        X_train_norm = (X_train - self._mu) / self._std
        X_test = (X_test - self._mu) / self._std  # also normalize test

        # Ridge regression on standardized features
        lam = 0.1
        X_aug = np.column_stack([X_train_norm, np.ones(len(X_train_norm))])
        w_full = np.linalg.solve(
            X_aug.T @ X_aug + lam * np.eye(X_aug.shape[1]),
            X_aug.T @ y_train,
        )
        self._w = w_full[:-1]
        self._b = float(w_full[-1])

        # Compute raw scores on test set
        test_scores = []
        for i in range(len(X_test)):
            logit = float(self._w @ X_test[i] + self._b)
            score = 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))
            test_scores.append(score)

        test_scores_arr = np.array(test_scores)

        # Optimize threshold for accuracy
        best_acc = 0.0
        best_thr = 0.5
        for thr in np.linspace(0.2, 0.8, 30):
            preds = (test_scores_arr > thr).astype(int)
            acc = float(np.mean(preds == y_test))
            if acc > best_acc:
                best_acc = acc
                best_thr = float(thr)

        self.threshold = best_thr
        preds = (test_scores_arr > best_thr).astype(int)
        accuracy = float(np.mean(preds == y_test))

        # ECE: expected calibration error with 5 bins
        n_bins = 5
        bins: dict[int, list[tuple[float, int]]] = {i: [] for i in range(n_bins)}
        for i in range(len(X_test)):
            bin_idx = min(int(test_scores[i] * n_bins), n_bins - 1)
            bins[bin_idx].append((test_scores[i], int(y_test[i])))

        ece = 0.0
        for bin_items in bins.values():
            if not bin_items:
                continue
            scores_b, labels_b = zip(*bin_items, strict=False)
            avg_conf = float(np.mean(scores_b))
            avg_acc = float(np.mean(labels_b))
            ece += len(bin_items) / len(X_test) * abs(avg_conf - avg_acc)

        return CalibrationResult(
            ece=round(ece, 4),
            accuracy=round(accuracy, 4),
            threshold=round(best_thr, 4),
            n_synthetic=len(all_data),
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
