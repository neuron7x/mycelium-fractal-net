"""
Prometheus metrics for MyceliumFractalNet API.

Provides HTTP request metrics, latency histograms, simulation metrics, and a /metrics endpoint
for Prometheus scraping.

HTTP Metrics:
    mfn_http_requests_total: Counter of HTTP requests (labels: endpoint, method, status)
    mfn_http_request_duration_seconds: Histogram of request latency (labels: endpoint, method)
    mfn_http_requests_in_progress: Gauge of currently processing requests

Simulation Metrics (MFN-OBS-002):
    mfn_fractal_dimension: Histogram of fractal dimensions (labels: grid_size)
    mfn_growth_events_total: Counter of growth events (labels: grid_size, steps)
    mfn_simulation_duration_seconds: Histogram of simulation duration (labels: grid_size, steps)
    mfn_lyapunov_exponent: Gauge of Lyapunov exponent (most recent value)
    mfn_turing_activations_total: Counter of Turing pattern activations

Usage:
    from mycelium_fractal_net.integration.metrics import MetricsMiddleware, metrics_endpoint
    
    app.add_middleware(MetricsMiddleware)
    app.add_route("/metrics", metrics_endpoint)

Reference: docs/MFN_BACKLOG.md#MFN-OBS-001, MFN-OBS-002
"""

from __future__ import annotations

import time
from typing import Any, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response as StarletteResponse

from .api_config import MetricsConfig, get_api_config

# Try to import prometheus_client, provide fallback if not available
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; charset=utf-8"
    Counter = None  # type: ignore[misc,assignment]
    Histogram = None  # type: ignore[misc,assignment]
    Gauge = None  # type: ignore[misc,assignment]

    def generate_latest() -> bytes:  # type: ignore[misc]
        return b"# prometheus_client not installed\n"


# Define metrics (singleton pattern to avoid duplicate registration)
# Using a module-level dict to store created metrics

_METRICS_CREATED = False

# HTTP Metrics
REQUEST_COUNTER: Any = None
REQUEST_LATENCY: Any = None
REQUESTS_IN_PROGRESS: Any = None

# Simulation Metrics (MFN-OBS-002)
FRACTAL_DIMENSION: Any = None
GROWTH_EVENTS: Any = None
SIMULATION_DURATION: Any = None
LYAPUNOV_EXPONENT: Any = None
TURING_ACTIVATIONS: Any = None


def _create_metrics() -> None:
    """Create metrics if not already created."""
    global _METRICS_CREATED
    global REQUEST_COUNTER, REQUEST_LATENCY, REQUESTS_IN_PROGRESS
    global FRACTAL_DIMENSION, GROWTH_EVENTS, SIMULATION_DURATION
    global LYAPUNOV_EXPONENT, TURING_ACTIVATIONS

    if _METRICS_CREATED:
        return

    if PROMETHEUS_AVAILABLE:
        from prometheus_client import REGISTRY

        # HTTP Metrics
        # Check if metrics already exist in registry
        try:
            REQUEST_COUNTER = Counter(
                "mfn_http_requests_total",
                "Total number of HTTP requests",
                ["endpoint", "method", "status"],
            )
        except ValueError:
            # Already registered - get existing
            # Counter stores name without "_total" suffix internally
            metric_name = "mfn_http_requests"
            for collector in REGISTRY._names_to_collectors.values():
                if hasattr(collector, '_name') and collector._name == metric_name:
                    REQUEST_COUNTER = collector
                    break

        try:
            REQUEST_LATENCY = Histogram(
                "mfn_http_request_duration_seconds",
                "HTTP request latency in seconds",
                ["endpoint", "method"],
                buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            )
        except ValueError:
            # Already registered
            metric_name = "mfn_http_request_duration_seconds"
            for collector in REGISTRY._names_to_collectors.values():
                if hasattr(collector, '_name') and collector._name == metric_name:
                    REQUEST_LATENCY = collector
                    break

        try:
            REQUESTS_IN_PROGRESS = Gauge(
                "mfn_http_requests_in_progress",
                "Number of HTTP requests currently being processed",
                ["endpoint", "method"],
            )
        except ValueError:
            # Already registered
            metric_name = "mfn_http_requests_in_progress"
            for collector in REGISTRY._names_to_collectors.values():
                if hasattr(collector, '_name') and collector._name == metric_name:
                    REQUESTS_IN_PROGRESS = collector
                    break

        # =============================================================================
        # Simulation Metrics (MFN-OBS-002)
        # =============================================================================
        # Reference: docs/MFN_BACKLOG.md#MFN-OBS-002

        try:
            FRACTAL_DIMENSION = Histogram(
                "mfn_fractal_dimension",
                "Fractal dimension (box-counting) of simulation output",
                ["grid_size"],
                # Buckets optimized for D ∈ [1.0, 2.0] (typical range for 2D patterns)
                buckets=(1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.584, 1.6, 1.7, 1.8, 1.9, 2.0),
            )
        except ValueError:
            metric_name = "mfn_fractal_dimension"
            for collector in REGISTRY._names_to_collectors.values():
                if hasattr(collector, '_name') and collector._name == metric_name:
                    FRACTAL_DIMENSION = collector
                    break

        try:
            GROWTH_EVENTS = Counter(
                "mfn_growth_events_total",
                "Total growth events (spikes) during simulations",
                ["grid_size", "steps"],
            )
        except ValueError:
            metric_name = "mfn_growth_events"
            for collector in REGISTRY._names_to_collectors.values():
                if hasattr(collector, '_name') and collector._name == metric_name:
                    GROWTH_EVENTS = collector
                    break

        try:
            SIMULATION_DURATION = Histogram(
                "mfn_simulation_duration_seconds",
                "Duration of mycelium field simulation",
                ["grid_size", "steps"],
                # Buckets from 0.01s to 60s
                buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
            )
        except ValueError:
            metric_name = "mfn_simulation_duration_seconds"
            for collector in REGISTRY._names_to_collectors.values():
                if hasattr(collector, '_name') and collector._name == metric_name:
                    SIMULATION_DURATION = collector
                    break

        try:
            LYAPUNOV_EXPONENT = Gauge(
                "mfn_lyapunov_exponent",
                "Lyapunov exponent from most recent simulation (negative = stable)",
            )
        except ValueError:
            metric_name = "mfn_lyapunov_exponent"
            for collector in REGISTRY._names_to_collectors.values():
                if hasattr(collector, '_name') and collector._name == metric_name:
                    LYAPUNOV_EXPONENT = collector
                    break

        try:
            TURING_ACTIVATIONS = Counter(
                "mfn_turing_activations_total",
                "Total Turing morphogenesis pattern activations",
                ["grid_size"],
            )
        except ValueError:
            metric_name = "mfn_turing_activations"
            for collector in REGISTRY._names_to_collectors.values():
                if hasattr(collector, '_name') and collector._name == metric_name:
                    TURING_ACTIVATIONS = collector
                    break

    # When prometheus_client is not available, metrics stay as None
    # The middleware will handle None checks

    _METRICS_CREATED = True


# Initialize metrics on module load
_create_metrics()


# =============================================================================
# Simulation Metrics Recording Functions (MFN-OBS-002)
# =============================================================================


def record_simulation_metrics(
    grid_size: int,
    steps: int,
    fractal_dim: float,
    growth_events: int,
    duration_seconds: float,
    lyapunov_exp: Optional[float] = None,
    turing_activations: int = 0,
) -> None:
    """
    Record simulation-specific metrics.

    Called by simulation adapters to record metrics for Prometheus.

    Args:
        grid_size: Size of simulation grid (NxN).
        steps: Number of simulation steps.
        fractal_dim: Box-counting fractal dimension.
        growth_events: Number of growth events (spikes).
        duration_seconds: Simulation duration in seconds.
        lyapunov_exp: Optional Lyapunov exponent.
        turing_activations: Number of Turing pattern activations.
    """
    grid_str = str(grid_size)
    steps_str = str(steps)

    # Record fractal dimension histogram
    if FRACTAL_DIMENSION is not None:
        FRACTAL_DIMENSION.labels(grid_size=grid_str).observe(fractal_dim)

    # Record growth events counter
    if GROWTH_EVENTS is not None:
        GROWTH_EVENTS.labels(grid_size=grid_str, steps=steps_str).inc(growth_events)

    # Record simulation duration
    if SIMULATION_DURATION is not None:
        SIMULATION_DURATION.labels(grid_size=grid_str, steps=steps_str).observe(
            duration_seconds
        )

    # Record Lyapunov exponent
    if lyapunov_exp is not None and LYAPUNOV_EXPONENT is not None:
        LYAPUNOV_EXPONENT.set(lyapunov_exp)

    # Record Turing activations
    if turing_activations > 0 and TURING_ACTIVATIONS is not None:
        TURING_ACTIVATIONS.labels(grid_size=grid_str).inc(turing_activations)


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting HTTP request metrics.

    Records:
        - Request count (by endpoint, method, status)
        - Request latency (by endpoint, method)
        - In-progress requests (by endpoint, method)

    Attributes:
        config: Metrics configuration.
    """

    def __init__(
        self,
        app: Any,
        config: Optional[MetricsConfig] = None,
    ) -> None:
        """
        Initialize metrics middleware.

        Args:
            app: The ASGI application.
            config: Metrics configuration. If None, uses global config.
        """
        super().__init__(app)
        self.config = config or get_api_config().metrics

    def _normalize_endpoint(self, path: str) -> str:
        """
        Normalize endpoint path for metric labels.

        Removes path parameters to prevent label explosion.

        Args:
            path: Request path.

        Returns:
            str: Normalized endpoint path.
        """
        # Keep known endpoints as-is
        known_endpoints = [
            "/health",
            "/validate",
            "/simulate",
            "/nernst",
            "/federated/aggregate",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]

        for endpoint in known_endpoints:
            if path == endpoint or path.startswith(endpoint + "/"):
                return endpoint

        # For unknown paths, use a generic label
        return "/other"

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> StarletteResponse:
        """
        Process request and collect metrics.

        Args:
            request: Incoming request.
            call_next: Next middleware or route handler.

        Returns:
            Response: Route response.
        """
        if not self.config.enabled:
            return await call_next(request)

        endpoint = self._normalize_endpoint(request.url.path)
        method = request.method

        # Track in-progress requests
        if REQUESTS_IN_PROGRESS is not None:
            REQUESTS_IN_PROGRESS.labels(endpoint=endpoint, method=method).inc()
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            status_code = str(response.status_code)
        except Exception:
            status_code = "500"
            raise
        finally:
            # Record duration
            duration = time.perf_counter() - start_time
            if REQUEST_LATENCY is not None:
                REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(duration)

            # Decrement in-progress
            if REQUESTS_IN_PROGRESS is not None:
                REQUESTS_IN_PROGRESS.labels(endpoint=endpoint, method=method).dec()

            # Increment request counter
            if REQUEST_COUNTER is not None:
                REQUEST_COUNTER.labels(
                    endpoint=endpoint, method=method, status=status_code
                ).inc()

        return response


async def metrics_endpoint(request: Request) -> Response:
    """
    Endpoint handler for /metrics.

    Returns Prometheus-formatted metrics.

    Args:
        request: Incoming request.

    Returns:
        Response: Prometheus metrics in text format.
    """
    metrics_output = generate_latest()

    return Response(
        content=metrics_output,
        media_type=CONTENT_TYPE_LATEST,
    )


def is_prometheus_available() -> bool:
    """
    Check if prometheus_client is available.

    Returns:
        bool: True if prometheus_client is installed.
    """
    return PROMETHEUS_AVAILABLE


__all__ = [
    # HTTP Metrics Middleware
    "MetricsMiddleware",
    "metrics_endpoint",
    # HTTP Metrics Variables
    "REQUEST_COUNTER",
    "REQUEST_LATENCY",
    "REQUESTS_IN_PROGRESS",
    # Simulation Metrics Variables (MFN-OBS-002)
    "FRACTAL_DIMENSION",
    "GROWTH_EVENTS",
    "SIMULATION_DURATION",
    "LYAPUNOV_EXPONENT",
    "TURING_ACTIVATIONS",
    # Simulation Metrics Recording
    "record_simulation_metrics",
    # Utilities
    "is_prometheus_available",
]
