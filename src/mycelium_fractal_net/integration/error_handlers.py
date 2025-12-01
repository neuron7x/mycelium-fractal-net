"""
Standardized error handlers for MyceliumFractalNet API.

Provides consistent error response format across all API endpoints.
Implements exception handlers for common error types including:
- Validation errors (Pydantic)
- Authentication errors
- Rate limiting errors
- Internal server errors

Reference: docs/MFN_BACKLOG.md#MFN-API-005
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, cast

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .logging_config import get_logger
from .schemas import ErrorCode, ErrorDetail, ErrorResponse

logger = get_logger("error_handlers")


def create_error_response(
    error_code: str,
    message: str,
    details: Optional[List[ErrorDetail]] = None,
    request_id: Optional[str] = None,
) -> ErrorResponse:
    """
    Create a standardized error response.

    Args:
        error_code: Machine-readable error code from ErrorCode.
        message: Human-readable error message.
        details: Optional list of detailed error information.
        request_id: Optional request correlation ID.

    Returns:
        ErrorResponse with all fields populated.
    """
    return ErrorResponse(
        error_code=error_code,
        message=message,
        detail=message,  # For backward compatibility
        details=details,
        request_id=request_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _extract_request_id(request: Request) -> Optional[str]:
    """Extract request ID from request state or headers."""
    # Try request state first (set by RequestIDMiddleware)
    if hasattr(request.state, "request_id"):
        return cast(str, request.state.request_id)
    # Fall back to header
    header_value = request.headers.get("X-Request-ID")
    return str(header_value) if header_value is not None else None


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Handle Pydantic validation errors.

    Converts validation errors to standardized error response format.

    Args:
        request: FastAPI request object.
        exc: RequestValidationError from Pydantic.

    Returns:
        JSONResponse with 422 status code and error details.
    """
    request_id = _extract_request_id(request)

    details: List[ErrorDetail] = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error.get("loc", []))
        details.append(
            ErrorDetail(
                field=field if field else None,
                message=error.get("msg", "Validation error"),
                value=str(error.get("input", ""))[:100] if error.get("input") else None,
            )
        )

    error_response = create_error_response(
        error_code=ErrorCode.VALIDATION_ERROR,
        message=f"Validation failed: {len(details)} error(s)",
        details=details,
        request_id=request_id,
    )

    logger.warning(
        f"Validation error: {len(details)} field(s) invalid",
        extra={
            "request_id": request_id,
            "error_count": len(details),
            "path": request.url.path,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump(exclude_none=True),
    )


async def pydantic_validation_exception_handler(
    request: Request, exc: ValidationError
) -> JSONResponse:
    """
    Handle Pydantic ValidationError (not FastAPI RequestValidationError).

    Args:
        request: FastAPI request object.
        exc: ValidationError from Pydantic.

    Returns:
        JSONResponse with 422 status code and error details.
    """
    request_id = _extract_request_id(request)

    details: List[ErrorDetail] = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error.get("loc", []))
        details.append(
            ErrorDetail(
                field=field if field else None,
                message=error.get("msg", "Validation error"),
            )
        )

    error_response = create_error_response(
        error_code=ErrorCode.VALIDATION_ERROR,
        message=f"Validation failed: {len(details)} error(s)",
        details=details,
        request_id=request_id,
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump(exclude_none=True),
    )


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """
    Handle ValueError exceptions.

    Args:
        request: FastAPI request object.
        exc: ValueError exception.

    Returns:
        JSONResponse with 400 status code.
    """
    request_id = _extract_request_id(request)

    error_response = create_error_response(
        error_code=ErrorCode.INVALID_REQUEST,
        message=str(exc),
        request_id=request_id,
    )

    logger.warning(
        f"Value error: {exc}",
        extra={"request_id": request_id, "path": request.url.path},
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=error_response.model_dump(exclude_none=True),
    )


async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle unhandled exceptions.

    Args:
        request: FastAPI request object.
        exc: Any unhandled exception.

    Returns:
        JSONResponse with 500 status code.
    """
    request_id = _extract_request_id(request)

    # Log full exception for debugging (not exposed to client)
    logger.error(
        f"Internal error: {type(exc).__name__}: {exc}",
        extra={"request_id": request_id, "path": request.url.path},
        exc_info=True,
    )

    error_response = create_error_response(
        error_code=ErrorCode.INTERNAL_ERROR,
        message="An internal error occurred. Please try again later.",
        request_id=request_id,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(exclude_none=True),
    )


def register_error_handlers(app: FastAPI) -> None:
    """
    Register all error handlers with FastAPI application.

    Args:
        app: FastAPI application instance.
    """
    # Validation errors
    # Note: add_exception_handler expects generic Exception handler signature,
    # but correctly routes specific exception types to their handlers at runtime
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(ValidationError, pydantic_validation_exception_handler)  # type: ignore[arg-type]

    # Value errors (invalid business logic)
    app.add_exception_handler(ValueError, value_error_handler)  # type: ignore[arg-type]

    # Generic exception handler (must be last)
    app.add_exception_handler(Exception, internal_error_handler)


__all__ = [
    "create_error_response",
    "validation_exception_handler",
    "pydantic_validation_exception_handler",
    "value_error_handler",
    "internal_error_handler",
    "register_error_handlers",
]
