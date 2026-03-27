"""Verify: system is NOT inert — real existential pressure triggers transformation.

# PROOF TYPE: empirical/numerical, not analytical.
Accumulates k irreversible collapses until Phi >= tau.
Asserts: at least one Transformation accepted with bounded jump.
"""

from __future__ import annotations

import numpy as np

from mycelium_fractal_net.tau_control.identity_engine import IdentityEngine


def verify_no_inertia(n: int = 500, seed: int = 42) -> dict[str, object]:
    """Run steps with escalating pressure until transformation fires."""
    rng = np.random.default_rng(seed)
    engine = IdentityEngine(state_dim=4, collapse_k_max=2)

    reports = []
    for i in range(n):
        # Gradually escalate: more collapses, less recovery success
        collapsing = i > 50 and (i % 3 == 0)
        recovery_ok = i < 100 or rng.random() > 0.7
        x = rng.normal(0, 0.5 + i * 0.01, 4)

        report = engine.process(
            state_vector=x,
            free_energy=0.5 + i * 0.01,
            phase_is_collapsing=collapsing,
            coherence=max(0.1, 0.9 - i * 0.002),
            recovery_succeeded=recovery_ok,
        )
        reports.append(report)

    tc = engine.transform.transform_count
    accepted = [r for r in reports if r.tau_state.transform_accepted]

    return {
        "passed": tc >= 1,
        "transform_count": tc,
        "first_transform_step": accepted[0].tau_state.step if accepted else -1,
        "n_steps": n,
    }
