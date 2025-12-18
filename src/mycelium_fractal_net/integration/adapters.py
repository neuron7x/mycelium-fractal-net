"""
Adapters for MyceliumFractalNet integration layer.

Provides thin adapter functions that bridge between the integration layer
(schemas, service context) and the numerical core (model.py, core engines).

These adapters do NOT contain business logic - they only:
1. Convert between schema types and core types
2. Delegate to core functions
3. Convert results back to schema types

Reference: docs/ARCHITECTURE.md, docs/MFN_SYSTEM_ROLE.md
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from mycelium_fractal_net import (
    ValidationConfig,
    compute_nernst_potential,
    estimate_fractal_dimension,
    run_validation,
    simulate_mycelium_field,
)
from mycelium_fractal_net.model import HierarchicalKrumAggregator

from .schemas import (
    FederatedAggregateRequest,
    FederatedAggregateResponse,
    NernstRequest,
    NernstResponse,
    SimulateRequest,
    SimulateResponse,
    ValidateRequest,
    ValidateResponse,
)
from .service_context import ServiceContext


def run_validation_adapter(
    request: ValidateRequest,
    ctx: ServiceContext,
) -> ValidateResponse:
    """
    Run validation cycle using the core validation function.

    Adapts ValidateRequest to ValidationConfig, runs validation,
    and converts results to ValidateResponse.

    Args:
        request: Validation request with parameters.
        ctx: Service context (used for mode tracking, metadata).

    Returns:
        ValidateResponse: Validation results with loss metrics.

    Raises:
        ValueError: If validation parameters are invalid.
        RuntimeError: If validation fails.
    """
    # Convert request to ValidationConfig
    cfg = ValidationConfig(
        seed=request.seed,
        epochs=request.epochs,
        batch_size=request.batch_size,
        grid_size=request.grid_size,
        steps=request.steps,
        turing_enabled=request.turing_enabled,
        quantum_jitter=request.quantum_jitter,
    )

    # Run validation using core function
    metrics: Dict[str, Any] = run_validation(cfg)

    # Convert to response
    return ValidateResponse(
        loss_start=metrics["loss_start"],
        loss_final=metrics["loss_final"],
        loss_drop=metrics["loss_drop"],
        pot_min_mV=metrics["pot_min_mV"],
        pot_max_mV=metrics["pot_max_mV"],
        example_fractal_dim=metrics["example_fractal_dim"],
        lyapunov_exponent=metrics["lyapunov_exponent"],
        growth_events=metrics["growth_events"],
        nernst_symbolic_mV=metrics["nernst_symbolic_mV"],
        nernst_numeric_mV=metrics["nernst_numeric_mV"],
    )


def run_simulation_adapter(
    request: SimulateRequest,
    ctx: ServiceContext,
) -> SimulateResponse:
    """
    Run field simulation using the core simulation function.

    Adapts SimulateRequest to core parameters, runs simulation,
    and converts results to SimulateResponse.

    Args:
        request: Simulation request with parameters.
        ctx: Service context (provides RNG if seed not in request).

    Returns:
        SimulateResponse: Simulation results with field statistics.

    Raises:
        ValueError: If simulation parameters are invalid.
        RuntimeError: If simulation fails.
    """
    # Use seed from request
    rng = np.random.default_rng(request.seed)

    # Run simulation using core function
    field, growth_events = simulate_mycelium_field(
        rng=rng,
        grid_size=request.grid_size,
        steps=request.steps,
        alpha=request.alpha,
        spike_probability=request.spike_probability,
        turing_enabled=request.turing_enabled,
    )

    # Compute fractal dimension
    binary = field > -0.060  # threshold -60 mV
    fractal_dim = estimate_fractal_dimension(binary)

    # Convert to response (field is in Volts, convert to mV)
    return SimulateResponse(
        growth_events=growth_events,
        pot_min_mV=float(field.min() * 1000.0),
        pot_max_mV=float(field.max() * 1000.0),
        pot_mean_mV=float(field.mean() * 1000.0),
        pot_std_mV=float(field.std() * 1000.0),
        fractal_dimension=fractal_dim,
    )


def compute_nernst_adapter(
    request: NernstRequest,
    ctx: ServiceContext,
) -> NernstResponse:
    """
    Compute Nernst potential using the core function.

    Adapts NernstRequest to core parameters, computes potential,
    and converts result to NernstResponse.

    Args:
        request: Nernst request with ion parameters.
        ctx: Service context (unused, for consistency).

    Returns:
        NernstResponse: Computed membrane potential in mV.

    Raises:
        ValueError: If ion concentrations are invalid.
    """
    # Compute Nernst potential using core function
    e_volts = compute_nernst_potential(
        z_valence=request.z_valence,
        concentration_out_molar=request.concentration_out_molar,
        concentration_in_molar=request.concentration_in_molar,
        temperature_k=request.temperature_k,
    )

    # Convert to response (Volts to mV)
    return NernstResponse(potential_mV=e_volts * 1000.0)


def aggregate_gradients_adapter(
    request: FederatedAggregateRequest,
    ctx: ServiceContext,
) -> FederatedAggregateResponse:
    """
    Aggregate gradients using Hierarchical Krum.

    Adapts FederatedAggregateRequest to core parameters, performs aggregation,
    and converts result to FederatedAggregateResponse.

    Args:
        request: Aggregation request with gradient vectors.
        ctx: Service context (unused, for consistency).

    Returns:
        FederatedAggregateResponse: Aggregated gradient vector.

    Raises:
        ValueError: If no gradients provided or gradients have inconsistent dimensions.
    """
    if not request.gradients:
        raise ValueError("No gradients provided")

    # Convert to tensors
    gradients = [torch.tensor(g, dtype=torch.float32) for g in request.gradients]

    # Create aggregator and run aggregation
    aggregator = HierarchicalKrumAggregator(
        num_clusters=request.num_clusters,
        byzantine_fraction=request.byzantine_fraction,
    )
    rng = ctx.rng
    result = aggregator.aggregate(gradients, rng=rng)

    # Convert to response
    return FederatedAggregateResponse(
        aggregated_gradient=result.tolist(),
        num_input_gradients=len(request.gradients),
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "run_validation_adapter",
    "run_simulation_adapter",
    "compute_nernst_adapter",
    "aggregate_gradients_adapter",
]
