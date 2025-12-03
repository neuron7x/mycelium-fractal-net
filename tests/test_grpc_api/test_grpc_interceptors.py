"""
Unit tests for gRPC interceptors.

Tests authentication, audit, and rate limiting interceptors.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import grpc
from mycelium_fractal_net.grpc.interceptors import (
    AuditInterceptor,
    AuthInterceptor,
    RateLimitInterceptor,
)


@pytest.fixture
def mock_handler_call_details():
    """Create mock handler call details."""
    details = MagicMock(spec=grpc.HandlerCallDetails)
    details.method = "/mfn.MFNFeaturesService/ExtractFeatures"
    details.invocation_metadata = []
    return details


@pytest.fixture
def mock_continuation():
    """Create mock continuation."""
    async def continuation(handler_call_details):
        handler = MagicMock()
        handler.unary_unary = AsyncMock(return_value="response")
        handler.request_deserializer = lambda x: x
        handler.response_serializer = lambda x: x
        return handler
    
    return continuation


class TestAuthInterceptor:
    """Tests for authentication interceptor."""
    
    @pytest.mark.asyncio
    async def test_auth_success(self, mock_handler_call_details, mock_continuation):
        """Test successful authentication."""
        # Setup valid auth metadata
        api_key = "test-key-123"
        request_id = "req-123"
        timestamp = str(time.time())
        
        # Calculate signature
        import hashlib
        import hmac
        message = f"{request_id}:{timestamp}"
        signature = hmac.new(
            api_key.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
        
        mock_handler_call_details.invocation_metadata = [
            ("x-api-key", api_key),
            ("x-request-id", request_id),
            ("x-timestamp", timestamp),
            ("x-signature", signature),
        ]
        
        with patch("mycelium_fractal_net.grpc.interceptors.get_api_config") as mock_config:
            mock_auth_config = MagicMock()
            mock_auth_config.api_keys = [api_key]
            mock_api_config = MagicMock()
            mock_api_config.auth = mock_auth_config
            mock_config.return_value = mock_api_config
            
            interceptor = AuthInterceptor()
            handler = await interceptor.intercept_service(
                mock_continuation,
                mock_handler_call_details,
            )
            
            assert handler is not None
    
    @pytest.mark.asyncio
    async def test_auth_missing_api_key(self, mock_handler_call_details, mock_continuation):
        """Test authentication failure with missing API key."""
        mock_handler_call_details.invocation_metadata = []
        
        with patch("mycelium_fractal_net.grpc.interceptors.get_api_config") as mock_config:
            mock_auth_config = MagicMock()
            mock_auth_config.api_keys = ["valid-key"]
            mock_api_config = MagicMock()
            mock_api_config.auth = mock_auth_config
            mock_config.return_value = mock_api_config
            
            interceptor = AuthInterceptor()
            handler = await interceptor.intercept_service(
                mock_continuation,
                mock_handler_call_details,
            )
            
            # Should return abort handler
            assert handler is not None
            # Handler should abort when called
            context = MagicMock()
            context.abort = AsyncMock()
            await handler.unary_unary(None, context)
            context.abort.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_auth_invalid_signature(self, mock_handler_call_details, mock_continuation):
        """Test authentication failure with invalid signature."""
        api_key = "test-key-123"
        request_id = "req-123"
        timestamp = str(time.time())
        
        mock_handler_call_details.invocation_metadata = [
            ("x-api-key", api_key),
            ("x-request-id", request_id),
            ("x-timestamp", timestamp),
            ("x-signature", "invalid-signature"),
        ]
        
        with patch("mycelium_fractal_net.grpc.interceptors.get_api_config") as mock_config:
            mock_auth_config = MagicMock()
            mock_auth_config.api_keys = [api_key]
            mock_api_config = MagicMock()
            mock_api_config.auth = mock_auth_config
            mock_config.return_value = mock_api_config
            
            interceptor = AuthInterceptor()
            handler = await interceptor.intercept_service(
                mock_continuation,
                mock_handler_call_details,
            )
            
            # Should return abort handler
            context = MagicMock()
            context.abort = AsyncMock()
            await handler.unary_unary(None, context)
            context.abort.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_auth_expired_timestamp(self, mock_handler_call_details, mock_continuation):
        """Test authentication failure with expired timestamp."""
        api_key = "test-key-123"
        request_id = "req-123"
        # Old timestamp (more than 5 minutes ago)
        timestamp = str(time.time() - 400)
        
        import hashlib
        import hmac
        message = f"{request_id}:{timestamp}"
        signature = hmac.new(
            api_key.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
        
        mock_handler_call_details.invocation_metadata = [
            ("x-api-key", api_key),
            ("x-request-id", request_id),
            ("x-timestamp", timestamp),
            ("x-signature", signature),
        ]
        
        with patch("mycelium_fractal_net.grpc.interceptors.get_api_config") as mock_config:
            mock_auth_config = MagicMock()
            mock_auth_config.api_keys = [api_key]
            mock_api_config = MagicMock()
            mock_api_config.auth = mock_auth_config
            mock_config.return_value = mock_api_config
            
            interceptor = AuthInterceptor()
            handler = await interceptor.intercept_service(
                mock_continuation,
                mock_handler_call_details,
            )
            
            # Should return abort handler
            context = MagicMock()
            context.abort = AsyncMock()
            await handler.unary_unary(None, context)
            context.abort.assert_called_once()


class TestAuditInterceptor:
    """Tests for audit logging interceptor."""
    
    @pytest.mark.asyncio
    async def test_audit_logs_request(self, mock_handler_call_details, mock_continuation):
        """Test that requests are logged."""
        mock_handler_call_details.invocation_metadata = [
            ("x-request-id", "req-123"),
        ]
        
        with patch("mycelium_fractal_net.grpc.interceptors.logger") as mock_logger:
            interceptor = AuditInterceptor()
            handler = await interceptor.intercept_service(
                mock_continuation,
                mock_handler_call_details,
            )
            
            # Should log start
            assert mock_logger.info.called
            
            # Execute handler
            context = MagicMock()
            await handler.unary_unary(None, context)
            
            # Should log completion
            assert mock_logger.info.call_count >= 2


class TestRateLimitInterceptor:
    """Tests for rate limiting interceptor."""
    
    @pytest.mark.asyncio
    async def test_rate_limit_within_limit(self, mock_handler_call_details, mock_continuation):
        """Test requests within rate limit."""
        mock_handler_call_details.invocation_metadata = [
            ("x-api-key", "test-key"),
            ("x-request-id", "req-123"),
        ]
        
        interceptor = RateLimitInterceptor(rps_limit=10, concurrent_limit=5)
        handler = await interceptor.intercept_service(
            mock_continuation,
            mock_handler_call_details,
        )
        
        # Should return normal handler
        assert handler is not None
        
        # Execute handler
        context = MagicMock()
        result = await handler.unary_unary(None, context)
        assert result == "response"
    
    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, mock_handler_call_details, mock_continuation):
        """Test rate limit exceeded."""
        mock_handler_call_details.invocation_metadata = [
            ("x-api-key", "test-key"),
            ("x-request-id", "req-123"),
        ]
        
        # Very low limit
        interceptor = RateLimitInterceptor(rps_limit=2, concurrent_limit=1)
        
        # First request should succeed
        await interceptor.intercept_service(
            mock_continuation,
            mock_handler_call_details,
        )
        
        # Second request should succeed
        await interceptor.intercept_service(
            mock_continuation,
            mock_handler_call_details,
        )
        
        # Third request should be rate limited
        handler3 = await interceptor.intercept_service(
            mock_continuation,
            mock_handler_call_details,
        )
        
        # Execute third handler (should abort)
        context = MagicMock()
        context.abort = AsyncMock()
        await handler3.unary_unary(None, context)
        
        # Should have aborted with RESOURCE_EXHAUSTED
        context.abort.assert_called_once()
        call_args = context.abort.call_args
        assert call_args[0][0] == grpc.StatusCode.RESOURCE_EXHAUSTED
    
    @pytest.mark.asyncio
    async def test_rate_limit_concurrent_release(
        self, mock_handler_call_details, mock_continuation
    ):
        """Test concurrent slot is released after request."""
        mock_handler_call_details.invocation_metadata = [
            ("x-api-key", "test-key"),
            ("x-request-id", "req-123"),
        ]
        
        interceptor = RateLimitInterceptor(rps_limit=100, concurrent_limit=1)
        
        # First request
        handler = await interceptor.intercept_service(
            mock_continuation,
            mock_handler_call_details,
        )
        
        context = MagicMock()
        await handler.unary_unary(None, context)
        
        # Concurrent count should be released
        assert interceptor._concurrent_counts["test-key"] == 0
