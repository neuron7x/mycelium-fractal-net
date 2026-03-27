"""
Meta-Core — Operational Protocol of Cognitive Engineering.

Implements the recursive pipeline:

    Reality_t = phi( S_Theta_t( h_t( R(A_t) ) ) )

Where:
    A_t     — agent state at time t
    R(A_t)  — recursive generation via MFN (reaction-diffusion + TDA + causal rules)
    h_t     — compression: FieldSequence -> MorphologyDescriptor (features)
    S_Theta — selection through GNC+ neuromodulatory vector Theta
    phi     — SovereignGate: 6-lens verification -> interface output

Theta = {alpha, rho, beta, tau, nu, sigma_E, sigma_U, lambda_pe, eta}

Reality is not given. It is generated as an interface output
of recursive agent dynamics, modulated by the latent vector Theta.

Consciousness is not a substance — it is a mode of attentional selection.
Agency is the computation, maintenance, and reconfiguration of boundary.
Spacetime is a compressed coordination format.

Ref: Vasylenko Y.O. (2026), Meta-Core. Myloradove, Ukraine.
     Friston (2010), doi:10.1038/nrn2787
     Hoffman (2019), The Case Against Reality
     Tononi (2004), doi:10.1186/1471-2202-5-42
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mycelium_fractal_net.types.field import FieldSequence


# ===================================================================
# Data structures
# ===================================================================

@dataclass
class AgentState:
    """
    A_t — agent state at time t.

    Contains the field (perceptual manifold), optional history,
    and agent-level metadata (goals, actions taken).
    """
    sequence: FieldSequence
    action_history: list[str] = field(default_factory=list)
    step: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionResult:
    """
    h_t(R(A_t)) — compressed representation after recursive generation.

    Contains morphology features, CCP triple, and anomaly detection.
    """
    D_f: float                       # fractal dimension
    phi_proxy: float                 # integrated information proxy
    R: float                         # phase coherence
    anomaly_score: float             # MFN anomaly [0, 1]
    regime: str                      # MFN regime label
    cognitive: bool                  # CCP cognitive state
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class SelectionResult:
    """
    S_Theta(h_t) — selection output after GNC+ theta modulation.

    Contains GNC+ diagnosis, meso-strategy, and modulated anomaly.
    """
    gnc_regime: str                  # optimal / hyperactivated / hypoactivated / dysregulated
    coherence: float                 # GNC+ coherence [0, 1]
    dominant_axis: str               # dominant neuromodulator
    meso_strategy: str               # EXPLORE / EXPLOIT / RESET
    theta_vector: NDArray[np.float64]  # 9-dim theta
    modulated_anomaly: float         # GNCBridge-modulated anomaly
    program_spine: dict[str, float]  # A_H, B_X, D_T
    ccp_gnc_consistent: bool         # CCP-GNC+ consistency


@dataclass
class RealityFrame:
    """
    Reality_t = phi(S_Theta(h_t(R(A_t))))

    The final interface output — a verified, agent-accessible
    configuration of experience at time t.
    """
    # Pipeline stages
    agent: AgentState
    compression: CompressionResult
    selection: SelectionResult

    # SovereignGate verification
    sovereign_passed: bool           # all 6 lenses passed
    sovereign_lenses: dict[str, bool]  # per-lens results
    sovereign_confidence: float      # aggregate confidence

    # Interface output
    reality_label: str               # cognitive / subcognitive / pathological / transitional
    reality_confidence: float        # [0, 1]
    theta_signature: str             # compact Theta representation

    def summary(self) -> str:
        """One-line reality summary."""
        return (
            f"Reality_t[step={self.agent.step}]: "
            f"{self.reality_label} (conf={self.reality_confidence:.3f}) | "
            f"D_f={self.compression.D_f:.3f} Phi={self.compression.phi_proxy:.3f} "
            f"R={self.compression.R:.3f} | "
            f"GNC+={self.selection.gnc_regime} coh={self.selection.coherence:.3f} "
            f"strategy={self.selection.meso_strategy} | "
            f"sovereign={'PASS' if self.sovereign_passed else 'FAIL'}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.agent.step,
            "reality_label": self.reality_label,
            "reality_confidence": self.reality_confidence,
            "compression": {
                "D_f": self.compression.D_f,
                "phi_proxy": self.compression.phi_proxy,
                "R": self.compression.R,
                "anomaly_score": self.compression.anomaly_score,
                "regime": self.compression.regime,
                "cognitive": self.compression.cognitive,
            },
            "selection": {
                "gnc_regime": self.selection.gnc_regime,
                "coherence": self.selection.coherence,
                "dominant_axis": self.selection.dominant_axis,
                "meso_strategy": self.selection.meso_strategy,
                "program_spine": self.selection.program_spine,
                "ccp_gnc_consistent": self.selection.ccp_gnc_consistent,
            },
            "sovereign": {
                "passed": self.sovereign_passed,
                "lenses": self.sovereign_lenses,
                "confidence": self.sovereign_confidence,
            },
            "theta_signature": self.theta_signature,
        }


# ===================================================================
# Pipeline stages
# ===================================================================

def R_generate(agent: AgentState) -> FieldSequence:
    """
    R(A_t) — Recursive generation.

    MFN reaction-diffusion engine produces field dynamics from agent state.
    This is identity when agent already carries a FieldSequence.
    In a live system, this would invoke simulate_history().
    """
    return agent.sequence


def h_compress(sequence: FieldSequence) -> CompressionResult:
    """
    h_t(R(A_t)) — Compression.

    FieldSequence -> {D_f, Phi, R, anomaly, regime, cognitive}.
    Combines MFN anomaly detection with CCP triple.
    """
    from mycelium_fractal_net.core.detect import detect_anomaly, detect_regime_shift
    from mycelium_fractal_net.analytics.ccp_metrics import compute_ccp_state

    anomaly = detect_anomaly(sequence)
    regime = detect_regime_shift(sequence)
    ccp = compute_ccp_state(sequence)

    return CompressionResult(
        D_f=ccp["D_f"],
        phi_proxy=ccp["phi_proxy"],
        R=ccp["R"],
        anomaly_score=anomaly.score,
        regime=regime.label,
        cognitive=ccp["cognitive"],
        features={
            "anomaly_confidence": anomaly.confidence,
            "regime_confidence": regime.confidence,
            "conditions_met": ccp["conditions_met"],
        },
    )


def S_select(
    compression: CompressionResult,
    gnc_levels: dict[str, float] | None = None,
) -> SelectionResult:
    """
    S_Theta(h_t) — Selection through neuromodulatory vector Theta.

    GNC+ processes compression output, applies Theta modulation,
    determines meso-strategy, checks CCP-GNC+ consistency.
    """
    from mycelium_fractal_net.neurochem.gnc import (
        GNCBridge,
        GNCState,
        MesoController,
        compute_gnc_state,
        gnc_diagnose,
    )
    from mycelium_fractal_net.neurochem.ccp_gnc_bridge import validate_ccp_gnc_consistency

    gnc_state = compute_gnc_state(gnc_levels)
    diag = gnc_diagnose(gnc_state)
    meso = MesoController()
    _, strategy = meso.step(gnc_state)

    bridge = GNCBridge(gnc_state)
    modulated = bridge.modulate_anomaly_score(compression.anomaly_score)

    ccp_dict = {
        "D_f": compression.D_f,
        "phi_proxy": compression.phi_proxy,
        "R": compression.R,
        "cognitive": compression.cognitive,
    }
    consistency = validate_ccp_gnc_consistency(ccp_dict, gnc_state)

    return SelectionResult(
        gnc_regime=diag.regime,
        coherence=diag.coherence,
        dominant_axis=diag.dominant_axis,
        meso_strategy=strategy,
        theta_vector=gnc_state.theta.copy(),
        modulated_anomaly=modulated,
        program_spine=diag.program_spine,
        ccp_gnc_consistent=consistency["consistent"],
    )


def phi_sovereign(
    compression: CompressionResult,
    selection: SelectionResult,
) -> tuple[bool, dict[str, bool], float]:
    """
    phi() — SovereignGate: 6-lens verification.

    Six lenses that must ALL pass for reality frame to be trusted:

    L1 (Bounds): All values within physical bounds
    L2 (Consistency): CCP-GNC+ states are consistent
    L3 (Falsifiability): GNC+ falsification conditions hold
    L4 (Coherence): GNC+ coherence above minimum threshold
    L5 (Cognitive): CCP cognitive window check
    L6 (Stability): Anomaly score below critical threshold

    Returns: (all_passed, per_lens, confidence)
    """
    from mycelium_fractal_net.neurochem.gnc import (
        GNCState,
        check_falsification,
        gnc_diagnose,
        compute_gnc_state,
        THETA_BOUNDS,
        LEVEL_BOUNDS,
    )

    lenses = {}

    # L1: Bounds — theta and anomaly within valid ranges
    theta_ok = bool(np.all(selection.theta_vector >= THETA_BOUNDS[0])
                     and np.all(selection.theta_vector <= THETA_BOUNDS[1]))
    anomaly_ok = 0.0 <= selection.modulated_anomaly <= 1.0
    lenses["L1_bounds"] = theta_ok and anomaly_ok

    # L2: Consistency — CCP and GNC+ agree
    lenses["L2_consistency"] = selection.ccp_gnc_consistent

    # L3: Falsifiability — GNC+ F1-F7 hold
    gnc_state = compute_gnc_state()  # rebuild for check
    gnc_state_real = GNCState(
        levels=np.full(7, 0.5),  # use actual theta from selection
        theta=selection.theta_vector.copy(),
    )
    diag = gnc_diagnose(gnc_state_real)
    falsification = check_falsification(gnc_state_real, diag)
    lenses["L3_falsifiability"] = all(falsification.values())

    # L4: Coherence — above 0.3 minimum
    lenses["L4_coherence"] = selection.coherence >= 0.3

    # L5: Cognitive — CCP triple check
    lenses["L5_cognitive"] = compression.cognitive

    # L6: Stability — modulated anomaly below 0.8
    lenses["L6_stability"] = selection.modulated_anomaly < 0.8

    all_passed = all(lenses.values())
    passed_count = sum(lenses.values())
    confidence = passed_count / len(lenses)

    return all_passed, lenses, confidence


def classify_reality(
    compression: CompressionResult,
    selection: SelectionResult,
    sovereign_passed: bool,
) -> tuple[str, float]:
    """
    Classify the reality frame into a label.

    Labels:
    - cognitive: all systems nominal, CCP satisfied, sovereign passed
    - subcognitive: stable but CCP not fully satisfied
    - transitional: some systems changing, meso in EXPLORE/RESET
    - pathological: sovereign failed or dysregulated
    """
    if not sovereign_passed:
        return "pathological", 0.2

    if compression.cognitive and selection.gnc_regime == "optimal":
        return "cognitive", min(0.95, selection.coherence)

    if selection.gnc_regime == "dysregulated":
        return "pathological", 0.3

    if selection.meso_strategy == "RESET":
        return "transitional", 0.5

    if compression.cognitive:
        return "cognitive", min(0.85, selection.coherence)

    return "subcognitive", 0.6


# ===================================================================
# Main pipeline
# ===================================================================

def compute_reality(
    agent: AgentState,
    gnc_levels: dict[str, float] | None = None,
) -> RealityFrame:
    """
    Reality_t = phi( S_Theta_t( h_t( R(A_t) ) ) )

    Full Meta-Core pipeline: one call from agent state to verified reality frame.

    Args:
        agent: AgentState with FieldSequence.
        gnc_levels: Optional GNC+ modulator levels. Default: all 0.5.

    Returns:
        RealityFrame — complete, verified interface output.
    """
    # R(A_t) — recursive generation
    sequence = R_generate(agent)

    # h_t — compression
    compression = h_compress(sequence)

    # S_Theta — selection
    selection = S_select(compression, gnc_levels)

    # phi — SovereignGate
    sovereign_passed, sovereign_lenses, sovereign_confidence = phi_sovereign(
        compression, selection
    )

    # Reality classification
    label, confidence = classify_reality(
        compression, selection, sovereign_passed
    )

    # Theta signature
    theta_sig = "|".join(f"{v:.2f}" for v in selection.theta_vector)

    return RealityFrame(
        agent=agent,
        compression=compression,
        selection=selection,
        sovereign_passed=sovereign_passed,
        sovereign_lenses=sovereign_lenses,
        sovereign_confidence=sovereign_confidence,
        reality_label=label,
        reality_confidence=confidence,
        theta_signature=theta_sig,
    )


def reality_trajectory(
    agent: AgentState,
    gnc_levels: dict[str, float] | None = None,
    steps: int | None = None,
) -> list[RealityFrame]:
    """
    Compute Reality_t for each step in agent history.

    If agent.sequence has history (T, N, N), computes one
    RealityFrame per timestep to track cognitive state evolution.
    """
    seq = agent.sequence
    if seq.history is None or steps == 1:
        return [compute_reality(agent, gnc_levels)]

    T = seq.history.shape[0]
    if steps is not None:
        T = min(T, steps)

    frames = []
    for t in range(T):
        frame_field = seq.history[t]
        frame_seq = FieldSequence(field=frame_field)
        frame_agent = AgentState(
            sequence=frame_seq,
            action_history=agent.action_history,
            step=t,
            metadata=agent.metadata,
        )
        frame = compute_reality(frame_agent, gnc_levels)
        frames.append(frame)

    return frames


# ===================================================================
# TESTS
# ===================================================================

def _run_tests() -> None:
    import traceback

    passed = 0
    failed = 0

    def _test(name, fn):
        nonlocal passed, failed
        try:
            fn()
            passed += 1
            print(f"  \u2713 {name}")
        except Exception as e:
            failed += 1
            print(f"  \u2717 {name}: {e}")

    print("=" * 65)
    print("Meta-Core Test Suite")
    print("Reality_t = phi( S_Theta( h_t( R(A_t) ) ) )")
    print("=" * 65)

    # --- Setup ---
    from mycelium_fractal_net.core.simulate import simulate_history
    from mycelium_fractal_net.types.field import SimulationSpec

    spec = SimulationSpec(grid_size=32, steps=32, seed=42)
    seq = simulate_history(spec)
    agent = AgentState(sequence=seq, step=0)

    # --- R(A_t) ---
    print("\n--- R(A_t): Recursive Generation ---")

    def _test_R_returns_sequence():
        result = R_generate(agent)
        assert isinstance(result, FieldSequence)
        assert result.field.shape == (32, 32)
    _test("R(A_t) returns FieldSequence", _test_R_returns_sequence)

    # --- h_t: Compression ---
    print("\n--- h_t: Compression ---")

    def _test_h_returns_compression():
        c = h_compress(seq)
        assert isinstance(c, CompressionResult)
        assert 0 <= c.D_f <= 3.0
        assert 0 <= c.R <= 1.0
        assert isinstance(c.cognitive, bool)
    _test("h_t returns CompressionResult", _test_h_returns_compression)

    def _test_h_anomaly_bounded():
        c = h_compress(seq)
        assert 0.0 <= c.anomaly_score <= 1.0
    _test("h_t anomaly in [0, 1]", _test_h_anomaly_bounded)

    # --- S_Theta: Selection ---
    print("\n--- S_Theta: Selection ---")

    def _test_S_returns_selection():
        c = h_compress(seq)
        s = S_select(c, {"Glutamate": 0.6, "GABA": 0.4})
        assert isinstance(s, SelectionResult)
        assert s.gnc_regime in ("optimal", "hyperactivated", "hypoactivated", "dysregulated")
        assert s.meso_strategy in ("EXPLORE", "EXPLOIT", "RESET")
    _test("S_Theta returns SelectionResult", _test_S_returns_selection)

    def _test_S_default_levels():
        c = h_compress(seq)
        s = S_select(c)
        assert s.coherence > 0
    _test("S_Theta works with default levels", _test_S_default_levels)

    def _test_S_theta_dim():
        c = h_compress(seq)
        s = S_select(c)
        assert s.theta_vector.shape == (9,)
    _test("S_Theta theta dimension = 9", _test_S_theta_dim)

    # --- phi: SovereignGate ---
    print("\n--- phi: SovereignGate (6 lenses) ---")

    def _test_phi_returns_6_lenses():
        c = h_compress(seq)
        s = S_select(c)
        ok, lenses, conf = phi_sovereign(c, s)
        assert len(lenses) == 6
        assert all(k.startswith("L") for k in lenses)
        assert 0.0 <= conf <= 1.0
    _test("phi returns 6 lenses", _test_phi_returns_6_lenses)

    def _test_phi_lens_names():
        c = h_compress(seq)
        s = S_select(c)
        _, lenses, _ = phi_sovereign(c, s)
        expected = {"L1_bounds", "L2_consistency", "L3_falsifiability",
                    "L4_coherence", "L5_cognitive", "L6_stability"}
        assert set(lenses.keys()) == expected
    _test("phi lens names correct", _test_phi_lens_names)

    # --- compute_reality: Full pipeline ---
    print("\n--- compute_reality: Full Pipeline ---")

    def _test_full_pipeline():
        rf = compute_reality(agent, {"Glutamate": 0.6, "GABA": 0.4})
        assert isinstance(rf, RealityFrame)
        assert rf.reality_label in ("cognitive", "subcognitive", "transitional", "pathological")
        assert 0.0 <= rf.reality_confidence <= 1.0
    _test("Full pipeline returns RealityFrame", _test_full_pipeline)

    def _test_full_pipeline_default():
        rf = compute_reality(agent)
        assert rf.reality_label in ("cognitive", "subcognitive", "transitional", "pathological")
    _test("Full pipeline with default GNC+", _test_full_pipeline_default)

    def _test_summary_not_empty():
        rf = compute_reality(agent)
        s = rf.summary()
        assert len(s) > 0
        assert "Reality_t" in s
    _test("summary() is not empty", _test_summary_not_empty)

    def _test_to_dict_keys():
        rf = compute_reality(agent)
        d = rf.to_dict()
        assert "reality_label" in d
        assert "compression" in d
        assert "selection" in d
        assert "sovereign" in d
        assert "theta_signature" in d
    _test("to_dict has all keys", _test_to_dict_keys)

    def _test_theta_signature_format():
        rf = compute_reality(agent)
        parts = rf.theta_signature.split("|")
        assert len(parts) == 9  # 9 theta params
    _test("theta_signature has 9 components", _test_theta_signature_format)

    # --- reality_trajectory ---
    print("\n--- reality_trajectory ---")

    def _test_trajectory_multi_step():
        frames = reality_trajectory(agent, steps=5)
        assert len(frames) == 5
        for f in frames:
            assert isinstance(f, RealityFrame)
    _test("trajectory with 5 steps", _test_trajectory_multi_step)

    def _test_trajectory_step_numbers():
        frames = reality_trajectory(agent, steps=3)
        assert [f.agent.step for f in frames] == [0, 1, 2]
    _test("trajectory step numbers correct", _test_trajectory_step_numbers)

    # --- classify_reality ---
    print("\n--- classify_reality ---")

    def _test_classify_cognitive():
        c = CompressionResult(D_f=1.7, phi_proxy=0.1, R=0.6,
                              anomaly_score=0.2, regime="stable", cognitive=True)
        s = SelectionResult(gnc_regime="optimal", coherence=0.8,
                            dominant_axis="Glutamate", meso_strategy="EXPLOIT",
                            theta_vector=np.full(9, 0.5),
                            modulated_anomaly=0.2, program_spine={},
                            ccp_gnc_consistent=True)
        label, conf = classify_reality(c, s, sovereign_passed=True)
        assert label == "cognitive"
    _test("classify: optimal+cognitive → cognitive", _test_classify_cognitive)

    def _test_classify_pathological():
        c = CompressionResult(D_f=1.0, phi_proxy=-0.5, R=0.1,
                              anomaly_score=0.9, regime="pathological_noise", cognitive=False)
        s = SelectionResult(gnc_regime="dysregulated", coherence=0.2,
                            dominant_axis="Opioid", meso_strategy="RESET",
                            theta_vector=np.full(9, 0.5),
                            modulated_anomaly=0.9, program_spine={},
                            ccp_gnc_consistent=False)
        label, conf = classify_reality(c, s, sovereign_passed=False)
        assert label == "pathological"
    _test("classify: dysregulated+failed → pathological", _test_classify_pathological)

    # --- AgentState ---
    print("\n--- AgentState ---")

    def _test_agent_state_fields():
        a = AgentState(sequence=seq, step=5, action_history=["observe", "adapt"])
        assert a.step == 5
        assert len(a.action_history) == 2
    _test("AgentState holds step and actions", _test_agent_state_fields)

    # --- Summary ---
    print("\n" + "=" * 65)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print("=" * 65)

    if failed == 0:
        print("\n--- DEMO: Full Reality Computation ---\n")
        rf = compute_reality(agent, {"Glutamate": 0.65, "GABA": 0.35, "Dopamine": 0.6})
        print(rf.summary())
        print()
        print("Sovereign lenses:")
        for lens, val in rf.sovereign_lenses.items():
            print(f"  {lens}: {'PASS' if val else 'FAIL'}")
        print(f"\nTheta = [{rf.theta_signature}]")

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_tests()
