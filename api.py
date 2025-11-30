"""
FastAPI server for MyceliumFractalNet v4.1.

Provides REST API for validation, simulation, and federated learning.
Uses the integration layer for consistent schema handling and service context.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health          - Health check
    POST /validate        - Run validation cycle
    POST /simulate        - Simulate mycelium field
    POST /nernst          - Compute Nernst potential
    POST /federated/aggregate - Aggregate gradients (Krum)
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

# Import schemas and adapters from integration layer
from mycelium_fractal_net.integration import (
    ExecutionMode,
    FederatedAggregateRequest,
    FederatedAggregateResponse,
    HealthResponse,
    NernstRequest,
    NernstResponse,
    ServiceContext,
    SimulateRequest,
    SimulateResponse,
    ValidateRequest,
    ValidateResponse,
    aggregate_gradients_adapter,
    compute_nernst_adapter,
    run_simulation_adapter,
    run_validation_adapter,
)

app = FastAPI(
    title="MyceliumFractalNet API",
    description="Bio-inspired adaptive network with fractal dynamics",
    version="4.1.0",
)


# Backward compatibility aliases for external consumers
# These re-export the integration layer schemas under the original names
ValidationRequest = ValidateRequest
ValidationResponse = ValidateResponse
SimulationRequest = SimulateRequest
SimulationResponse = SimulateResponse
AggregationRequest = FederatedAggregateRequest
AggregationResponse = FederatedAggregateResponse


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse()


@app.post("/validate", response_model=ValidateResponse)
async def validate(request: ValidateRequest) -> ValidateResponse:
    """Run validation cycle and return metrics."""
    try:
        ctx = ServiceContext(seed=request.seed, mode=ExecutionMode.API)
        return run_validation_adapter(request, ctx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate", response_model=SimulateResponse)
async def simulate(request: SimulateRequest) -> SimulateResponse:
    """Simulate mycelium field."""
    try:
        ctx = ServiceContext(seed=request.seed, mode=ExecutionMode.API)
        return run_simulation_adapter(request, ctx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/nernst", response_model=NernstResponse)
async def nernst(request: NernstRequest) -> NernstResponse:
    """Compute Nernst potential."""
    try:
        ctx = ServiceContext(mode=ExecutionMode.API)
        return compute_nernst_adapter(request, ctx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/federated/aggregate", response_model=FederatedAggregateResponse)
async def aggregate_gradients(
    request: FederatedAggregateRequest,
) -> FederatedAggregateResponse:
    """Aggregate gradients using hierarchical Krum."""
    try:
        ctx = ServiceContext(mode=ExecutionMode.API)
        return aggregate_gradients_adapter(request, ctx)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
