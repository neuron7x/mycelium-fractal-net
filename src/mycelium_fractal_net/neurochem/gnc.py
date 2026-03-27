"""General Neuromodulatory Control (GNC+) v2.0 — integrated into MFN.

General Causal Neuromodulation Platform:
A Control Architecture for Adaptive Intelligence.

Central Thesis:
    adaptive cognition = coupled neuromodulatory control over latent dynamics
    theta_{t+1} = f(theta_t, modulators_t, Omega, context_t, environment_t)

Seven axes (M):
    Glu  — Glutamate      — Plasticity
    GABA — GABA           — Stability
    NA   — Noradrenaline  — Salience
    5HT  — Serotonin      — Restraint
    DA   — Dopamine       — Reward
    ACh  — Acetylcholine  — Precision
    Op   — Opioid         — Resilience

Nine parameters Theta (Computational Core):
    alpha, rho, beta, tau, nu, sigma_E, sigma_U, lambda_pe, eta

Falsification conditions (F1-F7):
    F1: no stable mapping m_i -> Delta_Theta
    F3: Omega has no explanatory power beyond independent axes
    F5: Theta not recoverable from observations

Ref: Friston et al. (2012) Neural Comput 24:2201
     Schultz, Dayan & Montague (1997) Science 275:1593
     Dayan & Yu (2006) Neural Comput 18:1195
     Vasylenko (2026) General Causal Neuromodulation Platform
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "GNCBridge",
    "GNCDiagnosis",
    "GNCState",
    "MODULATORS",
    "ROLES",
    "SIGMA",
    "THETA",
    "compute_gnc_state",
    "gnc_diagnose",
    "step",
]

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

THETA = (
    "alpha", "rho", "beta", "tau", "nu",
    "sigma_E", "sigma_U", "lambda_pe", "eta",
)

THETA_DESC = {
    "alpha": "learning rate",
    "rho": "precision weighting",
    "beta": "policy stability",
    "tau": "inhibitory threshold",
    "nu": "reward valuation",
    "sigma_E": "expected uncertainty",
    "sigma_U": "unexpected uncertainty",
    "lambda_pe": "prediction-error valence",
    "eta": "effort persistence",
}

MODULATORS = (
    "Glutamate", "GABA", "Noradrenaline",
    "Serotonin", "Dopamine", "Acetylcholine", "Opioid",
)

ROLES = {
    "Glutamate": "Plasticity", "GABA": "Stability",
    "Noradrenaline": "Salience", "Serotonin": "Restraint",
    "Dopamine": "Reward", "Acetylcholine": "Precision",
    "Opioid": "Resilience",
}

# Sigma: sign structure d_theta_j / d_m_i
SIGMA: dict[str, dict[str, int]] = {
    "Glutamate":     {"alpha": +1, "rho": +1, "beta": 0, "tau": -1, "nu": 0, "sigma_E": 0, "sigma_U": 0, "lambda_pe": 0, "eta": 0},
    "GABA":          {"alpha": -1, "rho": +1, "beta": +1, "tau": +1, "nu": 0, "sigma_E": -1, "sigma_U": -1, "lambda_pe": 0, "eta": +1},
    "Noradrenaline": {"alpha": 0, "rho": 0, "beta": -1, "tau": 0, "nu": 0, "sigma_E": +1, "sigma_U": +1, "lambda_pe": 0, "eta": 0},
    "Serotonin":     {"alpha": 0, "rho": 0, "beta": +1, "tau": +1, "nu": -1, "sigma_E": 0, "sigma_U": 0, "lambda_pe": -1, "eta": +1},
    "Dopamine":      {"alpha": +1, "rho": 0, "beta": -1, "tau": -1, "nu": +1, "sigma_E": 0, "sigma_U": +1, "lambda_pe": +1, "eta": +1},
    "Acetylcholine": {"alpha": 0, "rho": +1, "beta": +1, "tau": 0, "nu": 0, "sigma_E": -1, "sigma_U": -1, "lambda_pe": 0, "eta": 0},
    "Opioid":        {"alpha": 0, "rho": 0, "beta": +1, "tau": 0, "nu": 0, "sigma_E": 0, "sigma_U": 0, "lambda_pe": 0, "eta": +1},
}

# Omega: pairwise interaction matrix (7x7)
_IDX = {m: i for i, m in enumerate(MODULATORS)}
_OMEGA = np.zeros((7, 7), dtype=np.float64)
_OMEGA[_IDX["Glutamate"], _IDX["GABA"]] = -0.6
_OMEGA[_IDX["GABA"], _IDX["Glutamate"]] = -0.6
_OMEGA[_IDX["Dopamine"], _IDX["Serotonin"]] = -0.4
_OMEGA[_IDX["Serotonin"], _IDX["Dopamine"]] = -0.4
_OMEGA[_IDX["Noradrenaline"], _IDX["Acetylcholine"]] = +0.3
_OMEGA[_IDX["Acetylcholine"], _IDX["Noradrenaline"]] = +0.3
_OMEGA[_IDX["Opioid"], :] = +0.2
_OMEGA[_IDX["Opioid"], _IDX["Opioid"]] = 0.0
_OMEGA[_IDX["Glutamate"], _IDX["Dopamine"]] = +0.3
_OMEGA[_IDX["Acetylcholine"], _IDX["Glutamate"]] = +0.2


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class GNCState:
    """Point x_t on the neuromodulatory manifold Psi. dim = 7 + 9 = 16."""

    modulators: dict[str, float]
    theta: dict[str, float]
    context: dict[str, float] = field(default_factory=dict)
    environment: dict[str, float] = field(default_factory=dict)

    @classmethod
    def default(cls) -> GNCState:
        return cls(
            modulators={m: 0.5 for m in MODULATORS},
            theta={t: 0.5 for t in THETA},
        )

    @classmethod
    def from_levels(cls, levels: dict[str, float], context: dict[str, float] | None = None) -> GNCState:
        modulators = {m: float(np.clip(levels.get(m, 0.5), 0.0, 1.0)) for m in MODULATORS}
        state = cls(modulators=modulators, theta={t: 0.5 for t in THETA}, context=context or {})
        state.theta = _compute_theta(state)
        return state

    def to_dict(self) -> dict[str, Any]:
        return {
            "modulators": {k: round(v, 4) for k, v in self.modulators.items()},
            "theta": {k: round(v, 4) for k, v in self.theta.items()},
        }

    def summary(self) -> str:
        lines = ["[GNC+ State]"]
        for m in MODULATORS:
            lv = self.modulators[m]
            dev = lv - 0.5
            bar = "+" if dev > 0.05 else ("-" if dev < -0.05 else "=")
            lines.append(f"  {bar} {m:15s} {lv:.2f} [{ROLES[m]}]")
        return "\n".join(lines)


@dataclass
class GNCDiagnosis:
    """Diagnostic result from GNC+ analysis."""

    regime: str
    coherence: float
    dominant_axis: str
    suppressed_axis: str
    theta_imbalance: float
    recommendation: str
    falsification_flags: list[str]

    def summary(self) -> str:
        lines = [
            f"[GNC+ Diagnosis] {self.regime}",
            f"  Coherence:  {self.coherence:.3f}",
            f"  Dominant:   {self.dominant_axis} [{ROLES[self.dominant_axis]}]",
            f"  Suppressed: {self.suppressed_axis} [{ROLES[self.suppressed_axis]}]",
            f"  Imbalance:  {self.theta_imbalance:.3f}",
        ]
        if self.falsification_flags:
            for f in self.falsification_flags:
                lines.append(f"  ! {f}")
        lines.append(f"  -> {self.recommendation}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# CORE COMPUTATION
# ═══════════════════════════════════════════════════════════════

def _compute_theta(state: GNCState) -> dict[str, float]:
    """Compute Theta from modulator levels via Phi + Sigma + Omega."""
    delta = {t: 0.0 for t in THETA}

    for m in MODULATORS:
        deviation = abs(state.modulators[m] - 0.5)
        sigma_row = SIGMA[m]
        for t in THETA:
            if sigma_row[t] != 0:
                delta[t] += sigma_row[t] * deviation * 0.3

    # Omega pairwise interactions
    levels = np.array([state.modulators[m] for m in MODULATORS])
    omega_effect = _OMEGA @ levels
    for i, m in enumerate(MODULATORS):
        targets = [t for t in THETA if SIGMA[m][t] != 0]
        for t in targets[:2]:
            delta[t] += omega_effect[i] * 0.05

    return {t: float(np.clip(0.5 + delta[t], 0.1, 0.9)) for t in THETA}


def compute_gnc_state(
    levels: dict[str, float] | None = None,
    context: dict[str, float] | None = None,
) -> GNCState:
    """Compute GNC+ state from modulator levels."""
    if levels is None:
        return GNCState.default()
    return GNCState.from_levels(levels, context)


def step(state: GNCState) -> GNCState:
    """One step on the neuromodulatory manifold: x_t -> x_{t+1}."""
    delta = {t: 0.0 for t in THETA}

    for m in MODULATORS:
        deviation = state.modulators[m] - 0.5
        sigma_row = SIGMA[m]
        for t in THETA:
            if sigma_row[t] != 0:
                delta[t] += sigma_row[t] * deviation * 0.3 * state.modulators[m]

    levels = np.array([state.modulators[m] for m in MODULATORS])
    omega_delta = _OMEGA @ levels
    for i, m in enumerate(MODULATORS):
        targets = [t for t in THETA if SIGMA[m][t] != 0]
        for t in targets[:2]:
            delta[t] += omega_delta[i] * 0.05

    rng = np.random.default_rng()
    eps = rng.normal(0, 0.01, len(THETA))
    theta_next = {
        t: float(np.clip(state.theta[t] + delta[t] + eps[i], 0.1, 0.9))
        for i, t in enumerate(THETA)
    }

    return GNCState(
        modulators=dict(state.modulators),
        theta=theta_next,
        context=dict(state.context),
        environment=dict(state.environment),
    )


def gnc_diagnose(state: GNCState) -> GNCDiagnosis:
    """Diagnose GNC+ state with falsification checks."""
    levels = np.array([state.modulators[m] for m in MODULATORS])
    deviations = levels - 0.5

    dominant_idx = int(np.argmax(np.abs(deviations)))
    suppressed_idx = int(np.argmin(levels))

    theta_arr = np.array([state.theta[t] for t in THETA])
    theta_imbalance = float(np.std(theta_arr))

    # Coherence: E/I balance x theta stability
    glu_dev = state.modulators["Glutamate"] - 0.5
    gaba_dev = -(state.modulators["GABA"] - 0.5)
    ei_balance = float(np.clip(1.0 - abs(glu_dev - gaba_dev), 0.0, 1.0))
    coherence = float(np.clip(ei_balance * (1.0 - theta_imbalance), 0.0, 1.0))

    mean_level = float(np.mean(levels))
    if coherence > 0.7 and theta_imbalance < 0.12:
        regime, rec = "optimal", "Maintain current state."
    elif mean_level > 0.72:
        regime, rec = "hyperactivated", "Consider GABA/Serotonin upregulation."
    elif mean_level < 0.28:
        regime, rec = "hypoactivated", "Consider Dopamine/Noradrenaline support."
    else:
        regime, rec = "dysregulated", "Check Glu/GABA and DA/5HT balance."

    # Falsification checks
    flags: list[str] = []
    for m in MODULATORS:
        sigma_row = SIGMA[m]
        for t in THETA:
            if sigma_row[t] != 0:
                mod_dev = state.modulators[m] - 0.5
                theta_dev = state.theta[t] - 0.5
                if abs(mod_dev) > 0.2 and abs(theta_dev) > 0.05:
                    if np.sign(theta_dev) != np.sign(sigma_row[t] * mod_dev):
                        flags.append(f"F1: {m}->{t} sign mismatch")

    omega_effect = float(np.linalg.norm(_OMEGA @ levels))
    if omega_effect < 0.05:
        flags.append("F3: Omega interaction near zero")

    if theta_imbalance < 0.01:
        flags.append("F5: Theta near-identical (not recoverable)")

    return GNCDiagnosis(
        regime=regime,
        coherence=coherence,
        dominant_axis=MODULATORS[dominant_idx],
        suppressed_axis=MODULATORS[suppressed_idx],
        theta_imbalance=theta_imbalance,
        recommendation=rec,
        falsification_flags=flags,
    )


# ═══════════════════════════════════════════════════════════════
# MFN BRIDGE
# ═══════════════════════════════════════════════════════════════

class GNCBridge:
    """Bridge between GNC+ and MFN pipeline.

    MFN diagnose -> updates Dopamine (reward prediction error)
    MFN auto_heal -> activates Opioid (resilience)
    GNC+ state -> modulates MFN anomaly score via NA/GABA
    """

    def __init__(self, state: GNCState | None = None) -> None:
        self.state = state or GNCState.default()

    def modulate_anomaly_score(self, raw_score: float) -> float:
        """NA boosts sensitivity, GABA dampens false positives."""
        na = self.state.modulators["Noradrenaline"]
        gaba = self.state.modulators["GABA"]
        boosted = raw_score * (1.0 + (na - 0.5) * 0.3) * (1.0 - (gaba - 0.5) * 0.2)
        return float(np.clip(boosted, 0.0, 1.0))

    def update_from_m_score(self, m_score: float) -> None:
        """MFN M-score -> Dopamine prediction error."""
        self.state.modulators["Dopamine"] = float(np.clip(m_score * 2.0, 0.0, 1.0))
        self.state.theta = _compute_theta(self.state)

    def activate_resilience(self) -> None:
        """auto_heal -> Opioid +0.2."""
        op = self.state.modulators["Opioid"]
        self.state.modulators["Opioid"] = float(np.clip(op + 0.2, 0.0, 1.0))
        self.state.theta = _compute_theta(self.state)

    def summary(self) -> str:
        return gnc_diagnose(self.state).summary()
