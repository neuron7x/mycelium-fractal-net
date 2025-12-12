"""
Event Logging Middleware for MyceliumFractalNet API.

Automatically logs all API requests and system events with full context.
Integrates with the system event logger to provide comprehensive audit trail.

Logged Events:
    - API requests (all endpoints)
    - Authentication attempts
    - File operations (configuration changes)
    - Data operations (fetch, merge, aggregate)

Features:
    - Automatic user identification from API key or request context
    - Request/response correlation
    - Duration tracking
    - Error logging
    - Structured event format

Reference: Problem statement - Логування всіх подій та дій системи
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from ..security.event_logger import EventStatus, EventType, get_event_logger
from .logging_config import get_request_id


class EventLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for automatic event logging.

    Logs all API requests and responses as system events for audit.
    """

    def __init__(self, app: Any) -> None:
        """
        Initialize event logging middleware.

        Args:
            app: The ASGI application.
        """
        super().__init__(app)
        self.event_logger = get_event_logger()

    def _get_user_id(self, request: Request) -> str:
        """
        Extract user ID from request.

        Args:
            request: Incoming request.

        Returns:
            User identifier (from API key, auth, or IP).
        """
        # Try to get from API key header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            # Use a hash or identifier from the key
            return f"api_key:{api_key[:8]}"

        # Try to get from authentication state
        if hasattr(request.state, "user_id"):
            return request.state.user_id

        # Fall back to client IP
        if request.client:
            return f"ip:{request.client.host}"

        return "anonymous"

    def _get_source_ip(self, request: Request) -> Optional[str]:
        """
        Get client IP address.

        Args:
            request: Incoming request.

        Returns:
            Client IP address or None.
        """
        if request.client:
            return request.client.host
        return None

    def _extract_files_from_request(self, request: Request) -> list[str]:
        """
        Extract file paths from request if applicable.

        Args:
            request: Incoming request.

        Returns:
            List of file paths mentioned in the request.
        """
        files = []

        # Check for file-related paths in URL
        if "/config" in request.url.path:
            files.append(request.url.path)

        # Could be extended to parse request body for file references
        return files

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process request and log events.

        Args:
            request: Incoming request.
            call_next: Next middleware or route handler.

        Returns:
            Response: Route response.
        """
        start_time = time.perf_counter()
        user_id = self._get_user_id(request)
        source_ip = self._get_source_ip(request)
        request_id = get_request_id()

        # Determine event type based on endpoint
        event_type = self._determine_event_type(request)

        # Extract metadata
        metadata = {
            "endpoint": request.url.path,
            "method": request.method,
            "query_params": dict(request.query_params),
        }

        # Extract files if applicable
        files_changed = self._extract_files_from_request(request)

        try:
            # Process request
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Determine status
            status = (
                EventStatus.SUCCESS
                if response.status_code < 400
                else EventStatus.FAILURE
            )

            # Add response info to metadata
            metadata["status_code"] = response.status_code
            metadata["duration_ms"] = round(duration_ms, 2)

            # Log the event
            self.event_logger.log_event(
                event_type=event_type,
                user_id=user_id,
                action_description=f"{request.method} {request.url.path}",
                status=status,
                files_changed=files_changed,
                metadata=metadata,
                request_id=request_id,
                source_ip=source_ip,
                duration_ms=round(duration_ms, 2),
            )

            return response

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log error event
            metadata["error"] = str(e)
            metadata["error_type"] = type(e).__name__

            self.event_logger.log_event(
                event_type=EventType.API_ERROR,
                user_id=user_id,
                action_description=f"Error: {request.method} {request.url.path}",
                status=EventStatus.FAILURE,
                files_changed=files_changed,
                metadata=metadata,
                request_id=request_id,
                source_ip=source_ip,
                duration_ms=round(duration_ms, 2),
            )

            raise

    def _determine_event_type(self, request: Request) -> EventType:
        """
        Determine event type based on request.

        Args:
            request: Incoming request.

        Returns:
            Appropriate EventType for this request.
        """
        path = request.url.path.lower()
        method = request.method.upper()

        # Authentication endpoints
        if "auth" in path or "login" in path or "key" in path:
            return EventType.AUTH_API_KEY

        # Data fetching
        if method == "GET" and ("data" in path or "fetch" in path):
            return EventType.FETCH

        # Configuration changes
        if "config" in path and method in ("POST", "PUT", "PATCH"):
            return EventType.COMMIT

        # Data aggregation/merging
        if "aggregate" in path or "merge" in path or "federated" in path:
            return EventType.MERGE

        # Pull/retrieve operations
        if method == "GET" and ("pull" in path or "retrieve" in path):
            return EventType.PULL_REQUEST

        # Push/publish operations
        if method == "POST" and ("push" in path or "publish" in path):
            return EventType.PUSH

        # Default to API request
        return EventType.API_REQUEST


__all__ = [
    "EventLoggingMiddleware",
]
