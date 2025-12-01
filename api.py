"""
FastAPI server for MyceliumFractalNet v4.1.

Provides REST API for validation, simulation, and federated learning.
Uses the integration layer for consistent schema handling and service context.

Production Features:
    - API key authentication (X-API-Key header)
    - Rate limiting (configurable per endpoint)
    - Prometheus metrics endpoint (/metrics)
    - Structured JSON logging with request IDs
    - Standardized error responses (MFN-API-005)

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health          - Health check (public)
    GET  /metrics         - Prometheus metrics (public)
    POST /validate        - Run validation cycle
    POST /simulate        - Simulate mycelium field
    POST /nernst          - Compute Nernst potential
    POST /federated/aggregate - Aggregate gradients (Krum)

Environment Variables:
    MFN_ENV              - Environment name: dev, staging, prod (default: dev)
    MFN_CORS_ORIGINS     - Comma-separated list of allowed CORS origins
    MFN_API_KEY_REQUIRED - Whether API key auth is required (default: false in dev)
    MFN_API_KEY          - Primary API key for authentication
    MFN_API_KEYS         - Comma-separated list of valid API keys
    MFN_RATE_LIMIT_REQUESTS - Max requests per minute (default: 100)
    MFN_RATE_LIMIT_ENABLED  - Enable rate limiting (default: false in dev)
    MFN_LOG_LEVEL        - Log level: DEBUG, INFO, WARNING, ERROR (default: INFO)
    MFN_LOG_FORMAT       - Log format: json or text (default: text in dev)
    MFN_METRICS_ENABLED  - Enable Prometheus metrics (default: true)

Reference: docs/MFN_BACKLOG.md#MFN-API-001, MFN-API-002, MFN-OBS-001, MFN-LOG-001, MFN-API-005
"""

from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

# Import schemas and adapters from integration layer
from mycelium_fractal_net.integration import (
    API_KEY_HEADER,
    REQUEST_ID_HEADER,
    APIKeyMiddleware,
    ExecutionMode,
    FederatedAggregateRequest,
    FederatedAggregateResponse,
    HealthResponse,
    MetricsMiddleware,
    NernstRequest,
    NernstResponse,
    RateLimitMiddleware,
    RequestIDMiddleware,
    RequestLoggingMiddleware,
    ServiceContext,
    SimulateRequest,
    SimulateResponse,
    ValidateRequest,
    ValidateResponse,
    aggregate_gradients_adapter,
    compute_nernst_adapter,
    get_api_config,
    get_logger,
    metrics_endpoint,
    register_error_handlers,
    run_simulation_adapter,
    run_validation_adapter,
    setup_logging,
)

# Initialize logging
setup_logging()
logger = get_logger("api")


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

# Get API configuration
api_config = get_api_config()

# Register standardized error handlers (MFN-API-005)
register_error_handlers(app)

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
        allow_headers=["*", API_KEY_HEADER],
        expose_headers=[REQUEST_ID_HEADER, "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    )

# Add production middleware (order matters: last added = first executed)
# 1. Request logging (outermost - logs all requests)
app.add_middleware(RequestLoggingMiddleware)

# 2. Request ID generation (needed for logging context)
app.add_middleware(RequestIDMiddleware)

# 3. Metrics collection
app.add_middleware(MetricsMiddleware)

# 4. Rate limiting
app.add_middleware(RateLimitMiddleware)

# 5. Authentication (innermost - first check)
app.add_middleware(APIKeyMiddleware)


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
    """Health check endpoint (public, no auth required)."""
    return HealthResponse()


@app.get("/metrics")
async def get_metrics(request: Request) -> Response:
    """
    Prometheus metrics endpoint (public, no auth required).

    Returns metrics in Prometheus text format including:
    - mfn_http_requests_total: Total HTTP requests
    - mfn_http_request_duration_seconds: Request latency histogram
    - mfn_http_requests_in_progress: Currently processing requests
    """
    return await metrics_endpoint(request)


@app.post("/validate", response_model=ValidateResponse)
async def validate(request: ValidateRequest) -> ValidateResponse:
    """Run validation cycle and return metrics."""
    ctx = ServiceContext(seed=request.seed, mode=ExecutionMode.API)
    return run_validation_adapter(request, ctx)


@app.post("/simulate", response_model=SimulateResponse)
async def simulate(request: SimulateRequest) -> SimulateResponse:
    """Simulate mycelium field."""
    ctx = ServiceContext(seed=request.seed, mode=ExecutionMode.API)
    return run_simulation_adapter(request, ctx)


@app.post("/nernst", response_model=NernstResponse)
async def nernst(request: NernstRequest) -> NernstResponse:
    """Compute Nernst potential."""
    ctx = ServiceContext(mode=ExecutionMode.API)
    return compute_nernst_adapter(request, ctx)


@app.post("/federated/aggregate", response_model=FederatedAggregateResponse)
async def aggregate_gradients(
    request: FederatedAggregateRequest,
) -> FederatedAggregateResponse:
    """Aggregate gradients using hierarchical Krum."""
    ctx = ServiceContext(mode=ExecutionMode.API)
    return aggregate_gradients_adapter(request, ctx)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
