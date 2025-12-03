"""
gRPC interceptors for authentication, audit, and rate limiting.

Implements async interceptors for:
    - Authentication (API key + HMAC signature verification)
    - Audit logging (request/response tracking)
    - Rate limiting (per-API-key RPS and concurrent limits)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from collections import defaultdict
from typing import Awaitable, Callable, Dict

from grpc.aio import ServerInterceptor

import grpc
from mycelium_fractal_net.integration import get_api_config, get_logger

logger = get_logger("grpc.interceptors")


class AuthInterceptor(ServerInterceptor):
    """
    Authentication interceptor for gRPC.
    
    Validates:
        - x-api-key: API key from auth config
        - x-signature: HMAC-SHA256 signature of request_id
        - x-timestamp: Request timestamp (max age: 5 minutes)
    """
    
    def __init__(self) -> None:
        """Initialize auth interceptor."""
        self.api_config = get_api_config()
    
    def _verify_signature(
        self,
        api_key: str,
        request_id: str,
        timestamp: str,
        signature: str,
    ) -> bool:
        """
        Verify HMAC-SHA256 signature.
        
        Args:
            api_key: API key used for signing
            request_id: Request ID
            timestamp: Request timestamp
            signature: Provided signature (hex)
            
        Returns:
            bool: True if signature is valid
        """
        message = f"{request_id}:{timestamp}"
        expected_sig = hmac.new(
            api_key.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
        
        # Constant-time comparison
        return hmac.compare_digest(signature, expected_sig)
    
    def _check_timestamp(self, timestamp: str) -> bool:
        """
        Check if timestamp is within acceptable range.
        
        Args:
            timestamp: ISO timestamp or unix timestamp
            
        Returns:
            bool: True if timestamp is recent (< 5 minutes old)
        """
        try:
            ts = float(timestamp)
            now = time.time()
            age = now - ts
            return 0 <= age <= 300  # 5 minutes
        except (ValueError, TypeError):
            return False
    
    async def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], Awaitable[grpc.RpcMethodHandler]],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """
        Intercept RPC call for authentication.
        
        Args:
            continuation: Next handler in chain
            handler_call_details: RPC call details
            
        Returns:
            RPC method handler or aborts with UNAUTHENTICATED
        """
        # Extract metadata
        metadata = dict(handler_call_details.invocation_metadata)
        
        api_key = metadata.get("x-api-key")
        signature = metadata.get("x-signature")
        timestamp = metadata.get("x-timestamp")
        request_id = metadata.get("x-request-id", "")
        
        # Check if API key is valid
        if not api_key or api_key not in self.api_config.auth.api_keys:
            logger.warning(
                f"Invalid API key for {handler_call_details.method}",
                extra={"request_id": request_id},
            )
            return self._abort_unauthenticated("Invalid or missing API key")
        
        # Check timestamp
        if not timestamp or not self._check_timestamp(timestamp):
            logger.warning(
                f"Invalid timestamp for {handler_call_details.method}",
                extra={"request_id": request_id},
            )
            return self._abort_unauthenticated("Invalid or expired timestamp")
        
        # Verify signature
        if not signature or not self._verify_signature(api_key, request_id, timestamp, signature):
            logger.warning(
                f"Invalid signature for {handler_call_details.method}",
                extra={"request_id": request_id},
            )
            return self._abort_unauthenticated("Invalid signature")
        
        # Authentication successful
        return await continuation(handler_call_details)
    
    def _abort_unauthenticated(self, details: str) -> grpc.RpcMethodHandler:
        """Create handler that aborts with UNAUTHENTICATED status."""
        async def abort(request, context):
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, details)
        
        return grpc.unary_unary_rpc_method_handler(
            abort,
            request_deserializer=lambda x: x,
            response_serializer=lambda x: x,
        )


class AuditInterceptor(ServerInterceptor):
    """
    Audit logging interceptor.
    
    Logs:
        - Request ID
        - Service and method name
        - Duration
        - Status (OK/ERROR)
    """
    
    async def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], Awaitable[grpc.RpcMethodHandler]],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """
        Intercept RPC call for audit logging.
        
        Args:
            continuation: Next handler in chain
            handler_call_details: RPC call details
            
        Returns:
            RPC method handler with logging
        """
        metadata = dict(handler_call_details.invocation_metadata)
        request_id = metadata.get("x-request-id", "unknown")
        method = handler_call_details.method
        
        start_time = time.time()
        
        logger.info(
            f"gRPC request started: {method}",
            extra={"request_id": request_id, "method": method},
        )
        
        # Continue with the call
        handler = await continuation(handler_call_details)
        
        # Wrap the handler to log completion
        if handler and handler.unary_unary:
            original_handler = handler.unary_unary
            
            async def wrapped_handler(request, context):
                try:
                    response = await original_handler(request, context)
                    duration = time.time() - start_time
                    logger.info(
                        f"gRPC request completed: {method}",
                        extra={
                            "request_id": request_id,
                            "method": method,
                            "duration_ms": int(duration * 1000),
                            "status": "OK",
                        },
                    )
                    return response
                except Exception as e:
                    duration = time.time() - start_time
                    logger.error(
                        f"gRPC request failed: {method}",
                        extra={
                            "request_id": request_id,
                            "method": method,
                            "duration_ms": int(duration * 1000),
                            "status": "ERROR",
                            "error": str(e),
                        },
                    )
                    raise
            
            return grpc.unary_unary_rpc_method_handler(
                wrapped_handler,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        
        return handler


class RateLimitInterceptor(ServerInterceptor):
    """
    Rate limiting interceptor.
    
    Limits:
        - RPS per API key
        - Concurrent requests per API key
    """
    
    def __init__(self, rps_limit: int = 1000, concurrent_limit: int = 50) -> None:
        """
        Initialize rate limiter.
        
        Args:
            rps_limit: Max requests per second per API key
            concurrent_limit: Max concurrent requests per API key
        """
        self.rps_limit = rps_limit
        self.concurrent_limit = concurrent_limit
        
        # Per-API-key state
        self._request_counts: Dict[str, list] = defaultdict(list)
        self._concurrent_counts: Dict[str, int] = defaultdict(int)
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
    
    async def _check_rate_limit(self, api_key: str) -> bool:
        """
        Check if rate limit is exceeded.
        
        Args:
            api_key: API key to check
            
        Returns:
            bool: True if within limits
        """
        async with self._locks[api_key]:
            now = time.time()
            
            # Clean old entries (older than 1 second)
            self._request_counts[api_key] = [
                ts for ts in self._request_counts[api_key]
                if now - ts < 1.0
            ]
            
            # Check RPS limit
            if len(self._request_counts[api_key]) >= self.rps_limit:
                return False
            
            # Check concurrent limit
            if self._concurrent_counts[api_key] >= self.concurrent_limit:
                return False
            
            # Record request
            self._request_counts[api_key].append(now)
            self._concurrent_counts[api_key] += 1
            
            return True
    
    async def _release(self, api_key: str) -> None:
        """Release concurrent request slot."""
        async with self._locks[api_key]:
            self._concurrent_counts[api_key] = max(
                0, self._concurrent_counts[api_key] - 1
            )
    
    async def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], Awaitable[grpc.RpcMethodHandler]],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """
        Intercept RPC call for rate limiting.
        
        Args:
            continuation: Next handler in chain
            handler_call_details: RPC call details
            
        Returns:
            RPC method handler or aborts with RESOURCE_EXHAUSTED
        """
        metadata = dict(handler_call_details.invocation_metadata)
        api_key = metadata.get("x-api-key", "default")
        request_id = metadata.get("x-request-id", "")
        
        # Check rate limit
        if not await self._check_rate_limit(api_key):
            logger.warning(
                f"Rate limit exceeded for {handler_call_details.method}",
                extra={"request_id": request_id, "api_key": api_key[:8] + "..."},
            )
            return self._abort_resource_exhausted("Rate limit exceeded")
        
        # Continue with the call
        handler = await continuation(handler_call_details)
        
        # Wrap handler to release slot on completion
        if handler and handler.unary_unary:
            original_handler = handler.unary_unary
            
            async def wrapped_handler(request, context):
                try:
                    return await original_handler(request, context)
                finally:
                    await self._release(api_key)
            
            return grpc.unary_unary_rpc_method_handler(
                wrapped_handler,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        
        return handler
    
    def _abort_resource_exhausted(self, details: str) -> grpc.RpcMethodHandler:
        """Create handler that aborts with RESOURCE_EXHAUSTED status."""
        async def abort(request, context):
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, details)
        
        return grpc.unary_unary_rpc_method_handler(
            abort,
            request_deserializer=lambda x: x,
            response_serializer=lambda x: x,
        )


__all__ = [
    "AuthInterceptor",
    "AuditInterceptor",
    "RateLimitInterceptor",
]
