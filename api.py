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

Environment Variables:
    MFN_CORS_ORIGINS     - Comma-separated list of allowed CORS origins
                           (default: "*" in development, empty in production)
    MFN_ENV              - Environment name: dev, staging, prod (default: dev)
"""

from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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


def _get_cors_origins() -> List[str]:
    """
    Get CORS origins from environment or defaults.

    In development (MFN_ENV=dev), allows all origins.
    In production (MFN_ENV=prod), requires explicit configuration via MFN_CORS_ORIGINS.

    Returns:
        List of allowed origin strings.
    """
    env = os.getenv("MFN_ENV", "dev").lower()
    cors_origins = os.getenv("MFN_CORS_ORIGINS", "")

    if cors_origins:
        # Explicit configuration takes precedence
        return [origin.strip() for origin in cors_origins.split(",") if origin.strip()]

    # Environment-based defaults
    if env == "dev":
        return ["*"]  # Allow all in development
    elif env == "staging":
        return ["http://localhost:3000", "http://localhost:8080"]
    else:
        # Production: no default origins; must be explicitly configured
        return []


app = FastAPI(
    title="MyceliumFractalNet API",
    description="Bio-inspired adaptive network with fractal dynamics",
    version="4.1.0",
)

# Configure CORS middleware
# Reference: docs/MFN_BACKLOG.md#MFN-API-003
_cors_origins = _get_cors_origins()
if _cors_origins:
    _allow_all = "*" in _cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if _allow_all else _cors_origins,
        allow_credentials=not _allow_all,  # Cannot use credentials with "*"
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
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
