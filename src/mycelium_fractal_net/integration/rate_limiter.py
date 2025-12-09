"""
Rate limiting middleware for MyceliumFractalNet API.

Implements in-memory rate limiting to prevent API abuse.
Uses a token bucket algorithm with configurable limits per endpoint.

**PRODUCTION WARNING**: In-memory rate limiting does not persist across restarts
and does NOT share state between multiple instances. This provides a false sense
of security in multi-replica deployments where each instance has its own rate limits.

For production deployments with multiple replicas (>1), you MUST:
- Set MFN_RATE_LIMIT_ENABLED=false and implement distributed rate limiting (Redis)
- OR accept that rate limits are per-instance (actual limit = config * num_replicas)
- Set MFN_RATE_LIMIT_WARN_MULTI_REPLICA=false to suppress startup warning

The middleware will log a warning if enabled in production without explicit acknowledgment.

Usage:
    from mycelium_fractal_net.integration.rate_limiter import RateLimitMiddleware
    
    middleware = RateLimitMiddleware(app, config)

Reference: docs/MFN_BACKLOG.md#MFN-API-002
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Optional, Tuple

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from .api_config import RateLimitConfig, get_api_config


@dataclass
class TokenBucket:
    """
    Token bucket for rate limiting.

    Implements the token bucket algorithm:
    - Tokens are added at a fixed rate
    - Requests consume tokens
    - If no tokens available, request is rejected

    Attributes:
        tokens: Current number of available tokens.
        last_update: Timestamp of last token update.
        max_tokens: Maximum tokens in bucket.
        refill_rate: Tokens per second refill rate.
    """

    tokens: float
    last_update: float
    max_tokens: int
    refill_rate: float

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume.

        Returns:
            bool: True if tokens were consumed, False if insufficient.
        """
        now = time.time()

        # Refill tokens based on elapsed time
        elapsed = now - self.last_update
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self.last_update = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True

        return False

    def time_until_available(self) -> float:
        """
        Calculate time until tokens will be available.

        Returns:
            float: Seconds until next token available.
        """
        if self.tokens >= 1:
            return 0.0

        tokens_needed = 1 - self.tokens
        return tokens_needed / self.refill_rate


class RateLimiter:
    """
    In-memory rate limiter using token buckets.

    Thread-safe implementation with per-client tracking.
    
    **WARNING**: Does NOT synchronize across multiple replicas.

    Attributes:
        config: Rate limit configuration.
        buckets: Per-client token buckets.
        _lock: Thread lock for bucket access.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None) -> None:
        """
        Initialize rate limiter.

        Args:
            config: Rate limit configuration. If None, uses global config.
        """
        self.config = config or get_api_config().rate_limit
        self.buckets: Dict[str, TokenBucket] = {}
        self._lock = Lock()
        
        # Warn if enabled in production without acknowledgment
        self._check_production_warning()

    def _check_production_warning(self) -> None:
        """Log warning if in-memory rate limiting is used in production."""
        if not self.config.enabled:
            return
        
        env = os.getenv("MFN_ENV", "prod").lower()
        suppress_warning = os.getenv("MFN_RATE_LIMIT_WARN_MULTI_REPLICA", "true").lower() == "false"
        
        if env in ("prod", "production") and not suppress_warning:
            import logging
            logger = logging.getLogger("rate_limiter")
            logger.warning(
                "In-memory rate limiting is enabled in production. "
                "This does NOT share state across replicas and provides weak protection "
                "in multi-instance deployments. Each replica has independent rate limits. "
                "For true distributed rate limiting, use Redis or disable with "
                "MFN_RATE_LIMIT_ENABLED=false. To suppress this warning, set "
                "MFN_RATE_LIMIT_WARN_MULTI_REPLICA=false."
            )

    def _get_client_key(self, request: Request) -> str:
        """
        Get a unique key for the client.

        Uses IP address as the primary identifier.
        SECURITY: Only trusts X-Forwarded-For if explicitly enabled via config.
        By default, uses direct client IP to prevent spoofing.

        Args:
            request: The incoming request.

        Returns:
            str: Client identifier key.
        """
        # Check if we should trust proxy headers
        # In production, this should only be enabled when behind a trusted reverse proxy
        trust_proxy = os.getenv("MFN_TRUST_PROXY_HEADERS", "false").lower() in ("true", "1", "yes")
        
        if trust_proxy:
            # Only when explicitly configured, check X-Real-IP first (more reliable)
            real_ip = request.headers.get("X-Real-IP", "")
            if real_ip:
                return real_ip.strip()
            
            # Fall back to X-Forwarded-For
            forwarded = request.headers.get("X-Forwarded-For", "")
            if forwarded:
                # Take the first IP in the chain (original client)
                return str(forwarded.split(",")[0].strip())

        # Default: use direct client IP (secure against spoofing)
        if request.client:
            return str(request.client.host)

        return "unknown"

    def _get_bucket_key(self, client: str, endpoint: str) -> str:
        """
        Create a bucket key from client and endpoint.

        Args:
            client: Client identifier.
            endpoint: Request endpoint path.

        Returns:
            str: Bucket key.
        """
        return f"{client}:{endpoint}"

    def _get_limit_for_endpoint(self, endpoint: str) -> int:
        """
        Get rate limit for a specific endpoint.

        Args:
            endpoint: Endpoint path.

        Returns:
            int: Requests per window for this endpoint.
        """
        # Check for specific endpoint limit
        for path, limit in self.config.per_endpoint_limits.items():
            if endpoint.startswith(path):
                return limit

        # Fall back to default
        return self.config.max_requests

    def check_rate_limit(
        self, request: Request
    ) -> Tuple[bool, int, int, Optional[int]]:
        """
        Check if request is within rate limit.

        Args:
            request: The incoming request.

        Returns:
            Tuple containing:
                - allowed: Whether request is allowed
                - limit: Current rate limit
                - remaining: Remaining requests in window
                - retry_after: Seconds until limit resets (if blocked)
        """
        if not self.config.enabled:
            return True, 0, 0, None

        endpoint = request.url.path
        client = self._get_client_key(request)
        bucket_key = self._get_bucket_key(client, endpoint)

        limit = self._get_limit_for_endpoint(endpoint)
        refill_rate = limit / self.config.window_seconds

        with self._lock:
            if bucket_key not in self.buckets:
                self.buckets[bucket_key] = TokenBucket(
                    tokens=float(limit),
                    last_update=time.time(),
                    max_tokens=limit,
                    refill_rate=refill_rate,
                )

            bucket = self.buckets[bucket_key]

            if bucket.consume():
                remaining = int(bucket.tokens)
                # Probabilistic cleanup to prevent unbounded memory growth
                self._maybe_cleanup()
                return True, limit, remaining, None
            else:
                retry_after = int(bucket.time_until_available()) + 1
                return False, limit, 0, retry_after

    def cleanup_expired(self, max_age_seconds: int = 3600) -> int:
        """
        Remove expired bucket entries to prevent memory leaks.

        Args:
            max_age_seconds: Maximum age before removal (default: 1 hour).

        Returns:
            int: Number of buckets removed.
        """
        now = time.time()
        removed = 0

        with self._lock:
            expired_keys = [
                key
                for key, bucket in self.buckets.items()
                if now - bucket.last_update > max_age_seconds
            ]

            for key in expired_keys:
                del self.buckets[key]
                removed += 1

        return removed

    def _maybe_cleanup(self) -> None:
        """
        Probabilistically trigger cleanup to prevent unbounded memory growth.
        
        Uses a 1% chance per call to trigger cleanup, which provides automatic
        eviction without requiring a background thread.
        """
        import random
        
        # 1% chance to trigger cleanup on any rate limit check
        if random.random() < 0.01:
            removed = self.cleanup_expired(max_age_seconds=3600)
            if removed > 0:
                import logging
                logger = logging.getLogger("rate_limiter")
                logger.debug(f"Cleaned up {removed} expired rate limit buckets")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting API requests.

    Applies rate limits based on client IP and endpoint.
    Returns 429 Too Many Requests when limits are exceeded.

    Rate Limit Headers:
        X-RateLimit-Limit: Maximum requests per window
        X-RateLimit-Remaining: Remaining requests in current window
        Retry-After: Seconds to wait before retrying (on 429)
    """

    def __init__(
        self,
        app: Any,
        config: Optional[RateLimitConfig] = None,
    ) -> None:
        """
        Initialize rate limit middleware.

        Args:
            app: The ASGI application.
            config: Static rate limit configuration. If None, uses global config
                   at request time (dynamic).
        """
        super().__init__(app)
        self._static_config = config
        self._limiter: Optional[RateLimiter] = None

    @property
    def config(self) -> RateLimitConfig:
        """Get the current rate limit config (dynamic lookup if not static)."""
        if self._static_config is not None:
            return self._static_config
        return get_api_config().rate_limit

    @property
    def limiter(self) -> RateLimiter:
        """Get the rate limiter instance."""
        if self._limiter is None:
            self._limiter = RateLimiter(self.config)
        return self._limiter

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process request and apply rate limiting.

        Args:
            request: Incoming request.
            call_next: Next middleware or route handler.

        Returns:
            Response: Route response or 429 Too Many Requests.
        """
        # Check rate limit (config is checked dynamically)
        allowed, limit, remaining, retry_after = self.limiter.check_rate_limit(request)

        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "error_code": "rate_limit_exceeded",
                    "retry_after": retry_after,
                },
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": str(retry_after),
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        if self.config.enabled:
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response


__all__ = [
    "TokenBucket",
    "RateLimiter",
    "RateLimitMiddleware",
]
