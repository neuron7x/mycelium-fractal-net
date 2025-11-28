"""
FastAPI server for MyceliumFractalNet v4.1.

Provides REST API for validation, simulation, and federated learning.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health          - Health check
    POST /validate        - Run validation cycle
    POST /simulate        - Simulate mycelium field
    POST /federated/aggregate - Aggregate gradients (Krum)
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mycelium_fractal_net import (
    ValidationConfig,
    compute_nernst_potential,
    estimate_fractal_dimension,
    run_validation,
    simulate_mycelium_field,
)
from mycelium_fractal_net.model import HierarchicalKrumAggregator

app = FastAPI(
    title="MyceliumFractalNet API",
    description="Bio-inspired adaptive network with fractal dynamics",
    version="4.1.0",
)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "4.1.0"


class ValidationRequest(BaseModel):
    """Validation request parameters."""

    seed: int = Field(default=42, ge=0)
    epochs: int = Field(default=1, ge=1, le=100)
    batch_size: int = Field(default=4, ge=1, le=64)
    grid_size: int = Field(default=64, ge=8, le=256)
    steps: int = Field(default=64, ge=1, le=1000)
    turing_enabled: bool = True
    quantum_jitter: bool = False


class ValidationResponse(BaseModel):
    """Validation response with metrics."""

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


class SimulationRequest(BaseModel):
    """Simulation request parameters."""

    seed: int = Field(default=42, ge=0)
    grid_size: int = Field(default=64, ge=8, le=256)
    steps: int = Field(default=64, ge=1, le=1000)
    alpha: float = Field(default=0.18, ge=0.0, le=1.0)
    spike_probability: float = Field(default=0.25, ge=0.0, le=1.0)
    turing_enabled: bool = True


class SimulationResponse(BaseModel):
    """Simulation response."""

    growth_events: int
    pot_min_mV: float
    pot_max_mV: float
    pot_mean_mV: float
    pot_std_mV: float
    fractal_dimension: float


class NernstRequest(BaseModel):
    """Nernst potential request."""

    z_valence: int = Field(default=1, ge=1, le=3)
    concentration_out_molar: float = Field(gt=0)
    concentration_in_molar: float = Field(gt=0)
    temperature_k: float = Field(default=310.0, ge=273.0, le=373.0)


class NernstResponse(BaseModel):
    """Nernst potential response."""

    potential_mV: float


class AggregationRequest(BaseModel):
    """Federated aggregation request."""

    gradients: List[List[float]]
    num_clusters: int = Field(default=10, ge=1, le=1000)
    byzantine_fraction: float = Field(default=0.2, ge=0.0, le=0.5)


class AggregationResponse(BaseModel):
    """Federated aggregation response."""

    aggregated_gradient: List[float]
    num_input_gradients: int


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse()


@app.post("/validate", response_model=ValidationResponse)
async def validate(request: ValidationRequest) -> Dict[str, Any]:
    """Run validation cycle and return metrics."""
    try:
        cfg = ValidationConfig(
            seed=request.seed,
            epochs=request.epochs,
            batch_size=request.batch_size,
            grid_size=request.grid_size,
            steps=request.steps,
            turing_enabled=request.turing_enabled,
            quantum_jitter=request.quantum_jitter,
        )
        metrics = run_validation(cfg)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate", response_model=SimulationResponse)
async def simulate(request: SimulationRequest) -> SimulationResponse:
    """Simulate mycelium field."""
    try:
        rng = np.random.default_rng(request.seed)
        field, growth_events = simulate_mycelium_field(
            rng=rng,
            grid_size=request.grid_size,
            steps=request.steps,
            alpha=request.alpha,
            spike_probability=request.spike_probability,
            turing_enabled=request.turing_enabled,
        )

        # Compute fractal dimension
        binary = field > -0.060
        D = estimate_fractal_dimension(binary)

        return SimulationResponse(
            growth_events=growth_events,
            pot_min_mV=float(field.min() * 1000.0),
            pot_max_mV=float(field.max() * 1000.0),
            pot_mean_mV=float(field.mean() * 1000.0),
            pot_std_mV=float(field.std() * 1000.0),
            fractal_dimension=D,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/nernst", response_model=NernstResponse)
async def nernst(request: NernstRequest) -> NernstResponse:
    """Compute Nernst potential."""
    try:
        e_v = compute_nernst_potential(
            z_valence=request.z_valence,
            concentration_out_molar=request.concentration_out_molar,
            concentration_in_molar=request.concentration_in_molar,
            temperature_k=request.temperature_k,
        )
        return NernstResponse(potential_mV=e_v * 1000.0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/federated/aggregate", response_model=AggregationResponse)
async def aggregate_gradients(request: AggregationRequest) -> AggregationResponse:
    """Aggregate gradients using hierarchical Krum."""
    try:
        # Convert to tensors
        gradients = [torch.tensor(g, dtype=torch.float32) for g in request.gradients]

        if len(gradients) == 0:
            raise HTTPException(status_code=400, detail="No gradients provided")

        aggregator = HierarchicalKrumAggregator(
            num_clusters=request.num_clusters,
            byzantine_fraction=request.byzantine_fraction,
        )

        result = aggregator.aggregate(gradients)

        return AggregationResponse(
            aggregated_gradient=result.tolist(),
            num_input_gradients=len(gradients),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
