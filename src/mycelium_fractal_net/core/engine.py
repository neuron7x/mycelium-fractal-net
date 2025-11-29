"""
High-level simulation engine for MyceliumFractalNet.

Provides the main entry point `run_mycelium_simulation` that runs a complete
mycelium field simulation and returns a structured SimulationResult.

This module integrates the numerical core from the ReactionDiffusionEngine
with the SimulationConfig/SimulationResult types.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .reaction_diffusion_engine import ReactionDiffusionConfig, ReactionDiffusionEngine
from .types import SimulationConfig, SimulationResult


def run_mycelium_simulation(config: SimulationConfig) -> SimulationResult:
    """
    Run a mycelium field simulation with the given configuration.

    This is the high-level API for running simulations. It initializes
    the field, runs the configured number of steps, and returns structured
    results including the final field state and optional history.

    Parameters
    ----------
    config : SimulationConfig
        Configuration parameters for the simulation including:
        - grid_size: Size of the 2D grid (N × N)
        - steps: Number of simulation steps
        - alpha: Diffusion coefficient (must satisfy CFL condition α ≤ 0.25)
        - spike_probability: Probability of spike events per step
        - turing_enabled: Enable Turing morphogenesis patterns
        - turing_threshold: Threshold for pattern activation
        - quantum_jitter: Enable quantum noise jitter
        - jitter_var: Variance of quantum jitter
        - seed: Random seed for reproducibility

    Returns
    -------
    SimulationResult
        Structured result containing:
        - field: Final 2D potential field in Volts (N × N)
        - history: Time series of field snapshots (T × N × N) if store_history=True
        - growth_events: Total number of growth events during simulation
        - metadata: Additional simulation metadata (timing, parameters, etc.)

    Examples
    --------
    >>> from mycelium_fractal_net import SimulationConfig, run_mycelium_simulation
    >>> config = SimulationConfig(grid_size=32, steps=10, seed=42)
    >>> result = run_mycelium_simulation(config)
    >>> result.field.shape
    (32, 32)
    >>> result.grid_size
    32

    Notes
    -----
    - The simulation uses periodic boundary conditions by default.
    - Field values are clamped to physiological bounds [-95 mV, +40 mV].
    - For reproducible results, always set the seed parameter.
    """
    # Validate config type
    if not isinstance(config, SimulationConfig):
        raise TypeError(f"config must be SimulationConfig, got {type(config).__name__}")

    start_time = time.perf_counter()

    # Create RD engine configuration from SimulationConfig
    rd_config = ReactionDiffusionConfig(
        grid_size=config.grid_size,
        alpha=config.alpha,
        turing_threshold=config.turing_threshold,
        quantum_jitter=config.quantum_jitter,
        jitter_var=config.jitter_var,
        spike_probability=config.spike_probability,
        random_seed=config.seed,
        check_stability=True,
    )

    # Create and run the engine
    engine = ReactionDiffusionEngine(rd_config)

    # Run simulation with history for tracking
    field, metrics = engine.simulate(
        steps=config.steps,
        turing_enabled=config.turing_enabled,
        return_history=False,
    )

    elapsed_time = time.perf_counter() - start_time

    # Build metadata
    metadata: dict[str, Any] = {
        "config": {
            "grid_size": config.grid_size,
            "steps": config.steps,
            "alpha": config.alpha,
            "spike_probability": config.spike_probability,
            "turing_enabled": config.turing_enabled,
            "turing_threshold": config.turing_threshold,
            "quantum_jitter": config.quantum_jitter,
            "jitter_var": config.jitter_var,
            "seed": config.seed,
        },
        "elapsed_time_s": elapsed_time,
        "steps_computed": metrics.steps_computed,
        "field_min_v": metrics.field_min_v,
        "field_max_v": metrics.field_max_v,
        "field_mean_v": metrics.field_mean_v,
        "field_std_v": metrics.field_std_v,
        "activator_mean": metrics.activator_mean,
        "inhibitor_mean": metrics.inhibitor_mean,
        "turing_activations": metrics.turing_activations,
        "clamping_events": metrics.clamping_events,
    }

    # Convert field to float64 for consistency
    final_field: NDArray[np.float64] = field.astype(np.float64)

    return SimulationResult(
        field=final_field,
        history=None,
        growth_events=metrics.growth_events,
        metadata=metadata,
    )


def run_mycelium_simulation_with_history(
    config: SimulationConfig,
) -> SimulationResult:
    """
    Run a mycelium field simulation with full history tracking.

    Similar to `run_mycelium_simulation` but stores the field state at each
    timestep, enabling analysis of temporal dynamics.

    Parameters
    ----------
    config : SimulationConfig
        Configuration parameters for the simulation.

    Returns
    -------
    SimulationResult
        Result with history array of shape (steps, grid_size, grid_size).

    Notes
    -----
    This function uses more memory than `run_mycelium_simulation` as it
    stores the complete field history. For large grids or many steps,
    consider using the base function instead.
    """
    if not isinstance(config, SimulationConfig):
        raise TypeError(f"config must be SimulationConfig, got {type(config).__name__}")

    start_time = time.perf_counter()

    # Create RD engine configuration
    rd_config = ReactionDiffusionConfig(
        grid_size=config.grid_size,
        alpha=config.alpha,
        turing_threshold=config.turing_threshold,
        quantum_jitter=config.quantum_jitter,
        jitter_var=config.jitter_var,
        spike_probability=config.spike_probability,
        random_seed=config.seed,
        check_stability=True,
    )

    engine = ReactionDiffusionEngine(rd_config)

    # Run simulation WITH history
    history_arr, metrics = engine.simulate(
        steps=config.steps,
        turing_enabled=config.turing_enabled,
        return_history=True,
    )

    elapsed_time = time.perf_counter() - start_time

    # Get final field from history
    final_field: NDArray[np.float64] = history_arr[-1].astype(np.float64)
    history: NDArray[np.float64] = history_arr.astype(np.float64)

    metadata: dict[str, Any] = {
        "config": {
            "grid_size": config.grid_size,
            "steps": config.steps,
            "alpha": config.alpha,
            "spike_probability": config.spike_probability,
            "turing_enabled": config.turing_enabled,
            "turing_threshold": config.turing_threshold,
            "quantum_jitter": config.quantum_jitter,
            "jitter_var": config.jitter_var,
            "seed": config.seed,
        },
        "elapsed_time_s": elapsed_time,
        "steps_computed": metrics.steps_computed,
        "field_min_v": metrics.field_min_v,
        "field_max_v": metrics.field_max_v,
        "field_mean_v": metrics.field_mean_v,
        "field_std_v": metrics.field_std_v,
        "activator_mean": metrics.activator_mean,
        "inhibitor_mean": metrics.inhibitor_mean,
        "turing_activations": metrics.turing_activations,
        "clamping_events": metrics.clamping_events,
    }

    return SimulationResult(
        field=final_field,
        history=history,
        growth_events=metrics.growth_events,
        metadata=metadata,
    )
