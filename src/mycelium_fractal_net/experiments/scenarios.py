"""Simulation scenarios for healthy vs pathological γ-scaling.

# EVIDENCE TYPE: real_simulation
# Healthy: Turing-enabled R-D with standard diffusion → γ ∈ [-7, -3], R² > 0.3
# Pathological: No Turing + low diffusion → γ ≈ 0 or R² ≈ 0 (no scaling)

Ref: Vasylenko (2026) γ-scaling hypothesis
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SCENARIO_HEALTHY", "SCENARIO_PATHOLOGICAL", "ScenarioConfig"]


@dataclass
class ScenarioConfig:
    name: str
    sim_params: dict[str, object]
    n_steps_base: int = 30
    n_steps_increment: int = 3
    n_sequences: int = 15
    n_runs: int = 5
    expected_gamma_range: tuple[float, float] = (-7.0, -3.0)
    description: str = ""


SCENARIO_HEALTHY = ScenarioConfig(
    name="healthy",
    sim_params={"grid_size": 32, "alpha": 0.18, "spike_probability": 0.25,
                "turing_enabled": True},
    expected_gamma_range=(-7.0, -3.0),
    description="Standard Turing R-D: clear γ-scaling, R² > 0.3",
)

SCENARIO_PATHOLOGICAL = ScenarioConfig(
    name="pathological",
    sim_params={"grid_size": 32, "alpha": 0.05, "spike_probability": 0.01,
                "turing_enabled": False},
    expected_gamma_range=(-2.0, 2.0),
    description="No Turing + low diffusion: weak or absent γ-scaling",
)
