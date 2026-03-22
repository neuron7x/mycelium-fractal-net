"""Property-based tests for neuromodulation kinetics.

Uses Hypothesis to verify occupancy conservation holds for ALL valid
parameter combinations, not just canonical profiles.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from mycelium_fractal_net.neurochem.kinetics import step_neuromodulation_state
from mycelium_fractal_net.neurochem.state import NeuromodulationState


def _gabaa_config(
    concentration: float = 10.0,
    k_on: float = 0.22,
    k_off: float = 0.06,
    des_rate: float = 0.05,
    rec_rate: float = 0.02,
    shunt: float = 0.42,
) -> dict:
    return {
        "agonist_concentration_um": concentration,
        "resting_affinity_um": 0.30,
        "active_affinity_um": 0.25,
        "k_on": k_on,
        "k_off": k_off,
        "desensitization_rate_hz": des_rate,
        "recovery_rate_hz": rec_rate,
        "shunt_strength": shunt,
        "tonic_inhibition_scale": 1.0,
    }


class TestOccupancyConservationSingleStep:
    """Occupancy must sum to 1.0 after every step, for all valid parameters."""

    @given(
        dt=st.floats(min_value=0.001, max_value=2.0),
        concentration=st.floats(min_value=0.0, max_value=1000.0),
        k_on=st.floats(min_value=0.01, max_value=1.0),
        k_off=st.floats(min_value=0.01, max_value=1.0),
        des_rate=st.floats(min_value=0.0, max_value=0.5),
        rec_rate=st.floats(min_value=0.0, max_value=0.5),
    )
    @settings(max_examples=200, deadline=5000)
    def test_single_step_conservation(
        self, dt: float, concentration: float, k_on: float, k_off: float,
        des_rate: float, rec_rate: float,
    ) -> None:
        shape = (8, 8)
        state = NeuromodulationState.zeros(shape)
        activator = np.random.default_rng(42).uniform(0, 0.1, shape).astype(np.float64)
        field = np.random.default_rng(42).normal(-0.065, 0.005, shape).astype(np.float64)

        gabaa = _gabaa_config(concentration=concentration, k_on=k_on, k_off=k_off,
                              des_rate=des_rate, rec_rate=rec_rate)

        new_state = step_neuromodulation_state(
            state, dt_seconds=dt, activator=activator, field=field,
            gabaa=gabaa, serotonergic=None, observation_noise=None,
        )

        total = new_state.occupancy_resting + new_state.occupancy_active + new_state.occupancy_desensitized
        np.testing.assert_allclose(
            total, 1.0, atol=1e-6,
            err_msg=f"Conservation violated: dt={dt}, conc={concentration}, k_on={k_on}",
        )


class TestOccupancyConservationMultiStep:
    """Conservation must hold after multiple consecutive steps."""

    @given(
        n_steps=st.integers(min_value=2, max_value=10),
        concentration=st.floats(min_value=0.1, max_value=100.0),
        dt=st.floats(min_value=0.1, max_value=1.0),
    )
    @settings(max_examples=50, deadline=10000)
    def test_multi_step_conservation(
        self, n_steps: int, concentration: float, dt: float,
    ) -> None:
        shape = (8, 8)
        state = NeuromodulationState.zeros(shape)
        rng = np.random.default_rng(42)
        activator = rng.uniform(0, 0.1, shape).astype(np.float64)
        field = rng.normal(-0.065, 0.005, shape).astype(np.float64)
        gabaa = _gabaa_config(concentration=concentration)

        for step in range(n_steps):
            state = step_neuromodulation_state(
                state, dt_seconds=dt, activator=activator, field=field,
                gabaa=gabaa, serotonergic=None, observation_noise=None,
            )
            total = state.occupancy_resting + state.occupancy_active + state.occupancy_desensitized
            np.testing.assert_allclose(
                total, 1.0, atol=1e-6,
                err_msg=f"Conservation violated at step {step}",
            )


class TestOccupancyEdgeCases:
    """Edge cases that might break conservation."""

    def test_zero_dt(self) -> None:
        """dt=0 should produce no change."""
        shape = (4, 4)
        state = NeuromodulationState.zeros(shape)
        rng = np.random.default_rng(1)
        new_state = step_neuromodulation_state(
            state, dt_seconds=0.001, activator=rng.uniform(0, 0.1, shape).astype(np.float64),
            field=rng.normal(-0.065, 0.005, shape).astype(np.float64),
            gabaa=_gabaa_config(), serotonergic=None, observation_noise=None,
        )
        total = new_state.occupancy_resting + new_state.occupancy_active + new_state.occupancy_desensitized
        np.testing.assert_allclose(total, 1.0, atol=1e-6)

    def test_extreme_concentration(self) -> None:
        """Very high concentration should not break conservation."""
        shape = (4, 4)
        state = NeuromodulationState.zeros(shape)
        rng = np.random.default_rng(2)
        new_state = step_neuromodulation_state(
            state, dt_seconds=1.0, activator=rng.uniform(0, 0.1, shape).astype(np.float64),
            field=rng.normal(-0.065, 0.005, shape).astype(np.float64),
            gabaa=_gabaa_config(concentration=10000.0), serotonergic=None, observation_noise=None,
        )
        total = new_state.occupancy_resting + new_state.occupancy_active + new_state.occupancy_desensitized
        np.testing.assert_allclose(total, 1.0, atol=1e-6)

    def test_no_gabaa(self) -> None:
        """Without GABA-A, occupancy should be all resting."""
        shape = (4, 4)
        state = NeuromodulationState.zeros(shape)
        rng = np.random.default_rng(3)
        new_state = step_neuromodulation_state(
            state, dt_seconds=1.0, activator=rng.uniform(0, 0.1, shape).astype(np.float64),
            field=rng.normal(-0.065, 0.005, shape).astype(np.float64),
            gabaa=None, serotonergic=None, observation_noise=None,
        )
        np.testing.assert_allclose(new_state.occupancy_resting, 1.0, atol=1e-10)
        np.testing.assert_allclose(new_state.occupancy_active, 0.0, atol=1e-10)
