"""
Pydantic schemas for MyceliumFractalNet API.

Provides unified request/response models for CLI, FastAPI, and experiments.
These schemas ensure consistent data validation and serialization across
all entry points (CLI, HTTP API, Python API).

Reference: docs/ARCHITECTURE.md, docs/MFN_SYSTEM_ROLE.md
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

# =============================================================================
# Health Check
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "4.1.0"


# =============================================================================
# Validation Cycle
# =============================================================================


class ValidateRequest(BaseModel):
    """
    Request parameters for validation cycle.

    Attributes:
        seed: Random seed for reproducibility. Range: [0, ∞). Default: 42.
        epochs: Number of training epochs. Range: [1, 100]. Default: 1.
        batch_size: Batch size for training. Range: [1, 64]. Default: 4.
        grid_size: Size of simulation grid (NxN). Range: [8, 256]. Default: 64.
        steps: Number of simulation steps. Range: [1, 1000]. Default: 64.
        turing_enabled: Enable Turing morphogenesis. Default: True.
        quantum_jitter: Enable quantum noise jitter. Default: False.
    """

    seed: int = Field(default=42, ge=0)
    epochs: int = Field(default=1, ge=1, le=100)
    batch_size: int = Field(default=4, ge=1, le=64)
    grid_size: int = Field(default=64, ge=8, le=256)
    steps: int = Field(default=64, ge=1, le=1000)
    turing_enabled: bool = True
    quantum_jitter: bool = False


class ValidateResponse(BaseModel):
    """
    Response from validation cycle.

    Contains loss metrics, potential statistics, and validation results.

    Attributes:
        loss_start: Initial loss value before training.
        loss_final: Final loss value after training.
        loss_drop: Absolute loss reduction (loss_start - loss_final).
        pot_min_mV: Minimum potential in millivolts.
        pot_max_mV: Maximum potential in millivolts.
        example_fractal_dim: Example fractal dimension from simulation.
        lyapunov_exponent: Lyapunov exponent (negative = stable).
        growth_events: Average growth events per simulation.
        nernst_symbolic_mV: Symbolic Nernst potential (mV).
        nernst_numeric_mV: Numeric Nernst potential (mV).
    """

    loss_start: float
    loss_final: float
    loss_drop: float
    pot_min_mV: float
    pot_max_mV: float
    example_fractal_dim: float
    lyapunov_exponent: float
    growth_events: float
    nernst_symbolic_mV: float
    nernst_numeric_mV: float


# =============================================================================
# Field Simulation
# =============================================================================


class SimulateRequest(BaseModel):
    """
    Request parameters for mycelium field simulation.

    Attributes:
        seed: Random seed for reproducibility. Range: [0, ∞). Default: 42.
        grid_size: Size of simulation grid (NxN). Range: [8, 256]. Default: 64.
        steps: Number of simulation steps. Range: [1, 1000]. Default: 64.
        alpha: Diffusion coefficient (CFL stability requires <= 0.25).
            Range: [0.0, 1.0]. Default: 0.18.
        spike_probability: Probability of growth events per step.
            Range: [0.0, 1.0]. Default: 0.25.
        turing_enabled: Enable Turing morphogenesis. Default: True.
    """

    seed: int = Field(default=42, ge=0)
    grid_size: int = Field(default=64, ge=8, le=256)
    steps: int = Field(default=64, ge=1, le=1000)
    alpha: float = Field(default=0.18, ge=0.0, le=1.0)
    spike_probability: float = Field(default=0.25, ge=0.0, le=1.0)
    turing_enabled: bool = True


class SimulateResponse(BaseModel):
    """
    Response from field simulation.

    Contains growth events, potential statistics, and fractal dimension.

    Attributes:
        growth_events: Number of growth events during simulation.
        pot_min_mV: Minimum potential in millivolts.
        pot_max_mV: Maximum potential in millivolts.
        pot_mean_mV: Mean potential in millivolts.
        pot_std_mV: Standard deviation of potential in millivolts.
        fractal_dimension: Box-counting fractal dimension (D ∈ [0, 2]).
    """

    growth_events: int
    pot_min_mV: float
    pot_max_mV: float
    pot_mean_mV: float
    pot_std_mV: float
    fractal_dimension: float


# =============================================================================
# Nernst Potential
# =============================================================================


class NernstRequest(BaseModel):
    """
    Request parameters for Nernst potential computation.

    Attributes:
        z_valence: Ion valence (K+=1, Ca2+=2, Cl-=1). Range: [1, 3]. Default: 1.
        concentration_out_molar: Extracellular concentration (mol/L). Must be > 0.
        concentration_in_molar: Intracellular concentration (mol/L). Must be > 0.
        temperature_k: Temperature in Kelvin. Range: [273, 373]. Default: 310 (37°C).
    """

    z_valence: int = Field(default=1, ge=1, le=3)
    concentration_out_molar: float = Field(gt=0)
    concentration_in_molar: float = Field(gt=0)
    temperature_k: float = Field(default=310.0, ge=273.0, le=373.0)


class NernstResponse(BaseModel):
    """
    Response from Nernst potential computation.

    Attributes:
        potential_mV: Computed membrane potential in millivolts.
    """

    potential_mV: float


# =============================================================================
# Federated Aggregation
# =============================================================================


class FederatedAggregateRequest(BaseModel):
    """
    Request parameters for federated gradient aggregation.

    Uses Hierarchical Krum aggregation for Byzantine-robust learning.

    Attributes:
        gradients: List of gradient vectors from federated clients.
            Each gradient is a list of float values.
        num_clusters: Number of clusters for hierarchical aggregation.
            Range: [1, 1000]. Default: 10.
        byzantine_fraction: Expected fraction of Byzantine (malicious) clients.
            Range: [0.0, 0.5]. Default: 0.2.
    """

    gradients: List[List[float]]
    num_clusters: int = Field(default=10, ge=1, le=1000)
    byzantine_fraction: float = Field(default=0.2, ge=0.0, le=0.5)


class FederatedAggregateResponse(BaseModel):
    """
    Response from federated aggregation.

    Attributes:
        aggregated_gradient: Aggregated gradient vector after Krum selection.
        num_input_gradients: Number of input gradients processed.
    """

    aggregated_gradient: List[float]
    num_input_gradients: int


# =============================================================================
# Error Response
# =============================================================================


class ErrorResponse(BaseModel):
    """
    Standard error response.

    Attributes:
        detail: Error message describing what went wrong.
        error_code: Optional machine-readable error code.
    """

    detail: str
    error_code: Optional[str] = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Health
    "HealthResponse",
    # Validation
    "ValidateRequest",
    "ValidateResponse",
    # Simulation
    "SimulateRequest",
    "SimulateResponse",
    # Nernst
    "NernstRequest",
    "NernstResponse",
    # Federated
    "FederatedAggregateRequest",
    "FederatedAggregateResponse",
    # Error
    "ErrorResponse",
]
