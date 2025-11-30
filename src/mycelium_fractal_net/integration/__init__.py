"""
Integration layer for MyceliumFractalNet.

Provides unified schemas, service context, and adapters for consistent
operation across CLI, HTTP API, and experiment entry points.

Components:
    - schemas: Pydantic models for request/response validation
    - service_context: Unified context with config, RNG, and engine handles
    - adapters: Thin bridge between integration layer and numerical core
    - api_config: Configuration management for API features
    - auth: API key authentication middleware
    - rate_limiter: Rate limiting middleware
    - metrics: Prometheus metrics collection
    - logging_config: Structured JSON logging

Usage:
    >>> from mycelium_fractal_net.integration import (
    ...     ValidateRequest,
    ...     ValidateResponse,
    ...     ServiceContext,
    ...     run_validation_adapter,
    ... )
    >>> ctx = ServiceContext(seed=42)
    >>> request = ValidateRequest(seed=42, epochs=1)
    >>> response = run_validation_adapter(request, ctx)

Reference: docs/ARCHITECTURE.md, docs/MFN_SYSTEM_ROLE.md
"""

from .adapters import (
    aggregate_gradients_adapter,
    compute_nernst_adapter,
    run_simulation_adapter,
    run_validation_adapter,
)
from .api_config import (
    APIConfig,
    AuthConfig,
    Environment,
    LoggingConfig,
    MetricsConfig,
    RateLimitConfig,
    get_api_config,
    reset_config,
)
from .auth import (
    API_KEY_HEADER,
    APIKeyMiddleware,
    require_api_key,
)
from .logging_config import (
    REQUEST_ID_HEADER,
    RequestIDMiddleware,
    RequestLoggingMiddleware,
    get_logger,
    get_request_id,
    set_request_id,
    setup_logging,
)
from .metrics import (
    MetricsMiddleware,
    is_prometheus_available,
    metrics_endpoint,
)
from .rate_limiter import (
    RateLimiter,
    RateLimitMiddleware,
)
from .schemas import (
    ErrorResponse,
    FederatedAggregateRequest,
    FederatedAggregateResponse,
    HealthResponse,
    NernstRequest,
    NernstResponse,
    SimulateRequest,
    SimulateResponse,
    ValidateRequest,
    ValidateResponse,
)
from .service_context import (
    ExecutionMode,
    ServiceContext,
    create_context_from_request,
)

__all__ = [
    # Schemas
    "HealthResponse",
    "ValidateRequest",
    "ValidateResponse",
    "SimulateRequest",
    "SimulateResponse",
    "NernstRequest",
    "NernstResponse",
    "FederatedAggregateRequest",
    "FederatedAggregateResponse",
    "ErrorResponse",
    # Service Context
    "ExecutionMode",
    "ServiceContext",
    "create_context_from_request",
    # Adapters
    "run_validation_adapter",
    "run_simulation_adapter",
    "compute_nernst_adapter",
    "aggregate_gradients_adapter",
    # API Configuration
    "Environment",
    "AuthConfig",
    "RateLimitConfig",
    "LoggingConfig",
    "MetricsConfig",
    "APIConfig",
    "get_api_config",
    "reset_config",
    # Authentication
    "API_KEY_HEADER",
    "APIKeyMiddleware",
    "require_api_key",
    # Rate Limiting
    "RateLimiter",
    "RateLimitMiddleware",
    # Metrics
    "MetricsMiddleware",
    "metrics_endpoint",
    "is_prometheus_available",
    # Logging
    "REQUEST_ID_HEADER",
    "RequestIDMiddleware",
    "RequestLoggingMiddleware",
    "setup_logging",
    "get_logger",
    "get_request_id",
    "set_request_id",
]
