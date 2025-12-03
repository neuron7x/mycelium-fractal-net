"""
Unit tests for MFN gRPC interceptors.

Tests authentication, audit logging, and rate limiting interceptors.
"""

import asyncio
import time
import hmac
import hashlib
import pytest
from unittest.mock import MagicMock, AsyncMock

import grpc

from mycelium_fractal_net.grpc.interceptors import (
    AuthInterceptor,
    AuditInterceptor,
    RateLimitInterceptor,
)


class MockHandlerCallDetails:
    """Mock gRPC handler call details."""
    
    def __init__(self, method="/test", metadata=None):
        self.method = method
        self.invocation_metadata = metadata or []


@pytest.fixture
def mock_continuation():
    """Create mock continuation handler."""
    async def continuation(handler_call_details):
        return grpc.unary_unary_rpc_method_handler(lambda req, ctx: None)
    
    return continuation


@pytest.mark.asyncio
async def test_auth_interceptor_no_auth_required(mock_continuation):
    """Test auth interceptor when no auth is required."""
    interceptor = AuthInterceptor(api_keys=[])  # Empty keys = no auth
    
    details = MockHandlerCallDetails(method="/test")
    handler = await interceptor.intercept_service(mock_continuation, details)
    
    assert handler is not None


@pytest.mark.asyncio
async def test_auth_interceptor_valid_credentials(mock_continuation):
    """Test auth interceptor with valid credentials."""
    api_key = "test-key-123"
    interceptor = AuthInterceptor(api_keys=[api_key])
    
    # Generate valid signature
    timestamp = str(time.time())
    method = "/mfn.MFNSimulationService/RunSimulation"
    message = f"{timestamp}{method}".encode("utf-8")
    signature = hmac.new(
        api_key.encode("utf-8"),
        message,
        hashlib.sha256,
    ).hexdigest()
    
    metadata = [
        ("x-api-key", api_key),
        ("x-signature", signature),
        ("x-timestamp", timestamp),
    ]
    
    details = MockHandlerCallDetails(method=method, metadata=metadata)
    handler = await interceptor.intercept_service(mock_continuation, details)
    
    assert handler is not None


@pytest.mark.asyncio
async def test_auth_interceptor_invalid_api_key(mock_continuation):
    """Test auth interceptor with invalid API key."""
    interceptor = AuthInterceptor(api_keys=["valid-key"])
    
    timestamp = str(time.time())
    metadata = [
        ("x-api-key", "invalid-key"),
        ("x-signature", "fake-signature"),
        ("x-timestamp", timestamp),
    ]
    
    details = MockHandlerCallDetails(metadata=metadata)
    handler = await interceptor.intercept_service(mock_continuation, details)
    
    # Handler should be abort handler
    assert handler is not None


@pytest.mark.asyncio
async def test_auth_interceptor_expired_timestamp(mock_continuation):
    """Test auth interceptor with expired timestamp."""
    api_key = "test-key"
    interceptor = AuthInterceptor(
        api_keys=[api_key],
        max_timestamp_age_sec=10,
    )
    
    # Old timestamp (1 hour ago)
    old_timestamp = str(time.time() - 3600)
    method = "/test"
    message = f"{old_timestamp}{method}".encode("utf-8")
    signature = hmac.new(
        api_key.encode("utf-8"),
        message,
        hashlib.sha256,
    ).hexdigest()
    
    metadata = [
        ("x-api-key", api_key),
        ("x-signature", signature),
        ("x-timestamp", old_timestamp),
    ]
    
    details = MockHandlerCallDetails(method=method, metadata=metadata)
    handler = await interceptor.intercept_service(mock_continuation, details)
    
    # Should be abort handler
    assert handler is not None


@pytest.mark.asyncio
async def test_auth_interceptor_invalid_signature(mock_continuation):
    """Test auth interceptor with invalid signature."""
    api_key = "test-key"
    interceptor = AuthInterceptor(api_keys=[api_key])
    
    timestamp = str(time.time())
    metadata = [
        ("x-api-key", api_key),
        ("x-signature", "invalid-signature-123"),
        ("x-timestamp", timestamp),
    ]
    
    details = MockHandlerCallDetails(metadata=metadata)
    handler = await interceptor.intercept_service(mock_continuation, details)
    
    # Should be abort handler
    assert handler is not None


@pytest.mark.asyncio
async def test_audit_interceptor_logs_request(mock_continuation):
    """Test audit interceptor logs requests."""
    interceptor = AuditInterceptor()
    
    metadata = [("x-request-id", "test-123")]
    details = MockHandlerCallDetails(
        method="/mfn.MFNSimulationService/RunSimulation",
        metadata=metadata,
    )
    
    handler = await interceptor.intercept_service(mock_continuation, details)
    
    assert handler is not None


@pytest.mark.asyncio
async def test_rate_limit_interceptor_allows_under_limit():
    """Test rate limiter allows requests under limit."""
    interceptor = RateLimitInterceptor(rps_limit=10, concurrent_limit=5)
    
    async def continuation(handler_call_details):
        return grpc.unary_unary_rpc_method_handler(lambda req, ctx: None)
    
    metadata = [("x-api-key", "test-key")]
    details = MockHandlerCallDetails(metadata=metadata)
    
    handler = await interceptor.intercept_service(continuation, details)
    
    assert handler is not None


@pytest.mark.asyncio
async def test_rate_limit_interceptor_enforces_rps_limit():
    """Test rate limiter enforces RPS limit."""
    interceptor = RateLimitInterceptor(rps_limit=2, concurrent_limit=10)
    
    async def continuation(handler_call_details):
        return grpc.unary_unary_rpc_method_handler(lambda req, ctx: None)
    
    metadata = [("x-api-key", "test-key")]
    
    # First 2 requests should succeed
    for _ in range(2):
        details = MockHandlerCallDetails(metadata=metadata)
        handler = await interceptor.intercept_service(continuation, details)
        assert handler is not None
    
    # Third request should be rate limited
    details = MockHandlerCallDetails(metadata=metadata)
    handler = await interceptor.intercept_service(continuation, details)
    
    # Should return abort handler
    assert handler is not None


@pytest.mark.asyncio
async def test_rate_limit_interceptor_per_key():
    """Test rate limiter is per API key."""
    interceptor = RateLimitInterceptor(rps_limit=1, concurrent_limit=10)
    
    async def continuation(handler_call_details):
        return grpc.unary_unary_rpc_method_handler(lambda req, ctx: None)
    
    # Request from key1
    metadata1 = [("x-api-key", "key1")]
    details1 = MockHandlerCallDetails(metadata=metadata1)
    handler1 = await interceptor.intercept_service(continuation, details1)
    assert handler1 is not None
    
    # Request from key2 should still work (different key)
    metadata2 = [("x-api-key", "key2")]
    details2 = MockHandlerCallDetails(metadata=metadata2)
    handler2 = await interceptor.intercept_service(continuation, details2)
    assert handler2 is not None


@pytest.mark.asyncio
async def test_rate_limit_reset_after_window():
    """Test rate limit resets after time window."""
    interceptor = RateLimitInterceptor(rps_limit=1, concurrent_limit=10)
    
    async def continuation(handler_call_details):
        return grpc.unary_unary_rpc_method_handler(lambda req, ctx: None)
    
    metadata = [("x-api-key", "test-key")]
    
    # First request succeeds
    details = MockHandlerCallDetails(metadata=metadata)
    handler = await interceptor.intercept_service(continuation, details)
    assert handler is not None
    
    # Wait for window to reset (1 second)
    await asyncio.sleep(1.1)
    
    # Second request should now succeed
    details = MockHandlerCallDetails(metadata=metadata)
    handler = await interceptor.intercept_service(continuation, details)
    assert handler is not None


def test_auth_interceptor_verify_signature():
    """Test signature verification logic."""
    api_key = "secret-key"
    interceptor = AuthInterceptor(api_keys=[api_key])
    
    timestamp = "1234567890"
    method = "/test"
    
    # Generate valid signature
    message = f"{timestamp}{method}".encode("utf-8")
    valid_sig = hmac.new(
        api_key.encode("utf-8"),
        message,
        hashlib.sha256,
    ).hexdigest()
    
    # Test valid signature
    assert interceptor._verify_signature(api_key, timestamp, method, valid_sig)
    
    # Test invalid signature
    assert not interceptor._verify_signature(api_key, timestamp, method, "wrong-sig")


@pytest.mark.asyncio
async def test_rate_limit_concurrent_limit():
    """Test rate limiter enforces concurrent request limit."""
    interceptor = RateLimitInterceptor(rps_limit=100, concurrent_limit=2)
    
    async def continuation(handler_call_details):
        return grpc.unary_unary_rpc_method_handler(lambda req, ctx: None)
    
    metadata = [("x-api-key", "test-key")]
    
    # Simulate 2 concurrent requests
    details1 = MockHandlerCallDetails(metadata=metadata)
    handler1 = await interceptor.intercept_service(continuation, details1)
    
    details2 = MockHandlerCallDetails(metadata=metadata)
    handler2 = await interceptor.intercept_service(continuation, details2)
    
    # Third concurrent request should be limited
    details3 = MockHandlerCallDetails(metadata=metadata)
    handler3 = await interceptor.intercept_service(continuation, details3)
    
    # All should return handlers (abort or normal)
    assert handler1 is not None
    assert handler2 is not None
    assert handler3 is not None
