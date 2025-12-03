"""
gRPC interceptors for MyceliumFractalNet.

Provides security, audit, and rate limiting for gRPC services.
All interceptors are async-compatible.

Reference: docs/MFN_GRPC_SPEC.md
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
from collections import defaultdict
from typing import Any, Callable, Optional

import grpc
from grpc.aio import ServerInterceptor

logger = logging.getLogger(__name__)


class AuthInterceptor(ServerInterceptor):
    """
    Authentication interceptor for gRPC.
    
    Validates API key, signature, and timestamp from metadata:
    - x-api-key: API key for authentication
    - x-signature: HMAC-SHA256 signature of request
    - x-timestamp: Request timestamp (Unix epoch)
    
    Signature is computed as: HMAC-SHA256(api_key, timestamp + method_name)
    """

    def __init__(
        self,
        api_keys: Optional[list[str]] = None,
        max_timestamp_age_sec: int = 300,
    ):
        """
        Initialize authentication interceptor.
        
        Args:
            api_keys: List of valid API keys (default: from env MFN_API_KEYS)
            max_timestamp_age_sec: Max age of timestamp in seconds (default: 300)
        """
        import os
        
        if api_keys is None:
            # Read from environment
            keys_str = os.getenv("MFN_API_KEYS", os.getenv("MFN_API_KEY", ""))
            api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        
        self.api_keys = set(api_keys) if api_keys else set()
        self.max_timestamp_age_sec = max_timestamp_age_sec
        self.require_auth = len(self.api_keys) > 0

    def _verify_signature(
        self,
        api_key: str,
        timestamp: str,
        method_name: str,
        signature: str,
    ) -> bool:
        """
        Verify HMAC signature.
        
        Args:
            api_key: API key
            timestamp: Request timestamp
            method_name: gRPC method name
            signature: Provided signature (hex)
            
        Returns:
            True if signature is valid
        """
        message = f"{timestamp}{method_name}".encode("utf-8")
        expected = hmac.new(
            api_key.encode("utf-8"),
            message,
            hashlib.sha256,
        ).hexdigest()
        
        # Constant-time comparison
        return hmac.compare_digest(signature, expected)

    async def intercept_service(
        self,
        continuation: Callable,
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """
        Intercept and authenticate requests.
        
        Args:
            continuation: Next handler in chain
            handler_call_details: Request details
            
        Returns:
            RPC method handler
        """
        # Skip auth if not required
        if not self.require_auth:
            return await continuation(handler_call_details)
        
        # Extract metadata
        metadata_dict = dict(handler_call_details.invocation_metadata)
        api_key = metadata_dict.get("x-api-key", "")
        signature = metadata_dict.get("x-signature", "")
        timestamp_str = metadata_dict.get("x-timestamp", "")
        
        # Validate API key
        if api_key not in self.api_keys:
            logger.warning(f"Invalid API key for {handler_call_details.method}")
            
            async def abort_unauthenticated(request, context):
                await context.abort(
                    grpc.StatusCode.UNAUTHENTICATED,
                    "Invalid API key",
                )
            
            return grpc.unary_unary_rpc_method_handler(abort_unauthenticated)
        
        # Validate timestamp
        try:
            timestamp = float(timestamp_str)
            now = time.time()
            age = abs(now - timestamp)
            
            if age > self.max_timestamp_age_sec:
                logger.warning(f"Expired timestamp for {handler_call_details.method}")
                
                async def abort_expired(request, context):
                    await context.abort(
                        grpc.StatusCode.UNAUTHENTICATED,
                        f"Timestamp too old: {age:.0f}s",
                    )
                
                return grpc.unary_unary_rpc_method_handler(abort_expired)
        
        except (ValueError, TypeError):
            logger.warning(f"Invalid timestamp for {handler_call_details.method}")
            
            async def abort_invalid_timestamp(request, context):
                await context.abort(
                    grpc.StatusCode.UNAUTHENTICATED,
                    "Invalid timestamp",
                )
            
            return grpc.unary_unary_rpc_method_handler(abort_invalid_timestamp)
        
        # Validate signature
        if not self._verify_signature(
            api_key, timestamp_str, handler_call_details.method, signature
        ):
            logger.warning(f"Invalid signature for {handler_call_details.method}")
            
            async def abort_invalid_signature(request, context):
                await context.abort(
                    grpc.StatusCode.UNAUTHENTICATED,
                    "Invalid signature",
                )
            
            return grpc.unary_unary_rpc_method_handler(abort_invalid_signature)
        
        # Auth successful
        return await continuation(handler_call_details)


class AuditInterceptor(ServerInterceptor):
    """
    Audit logging interceptor for gRPC.
    
    Logs all RPC calls with:
    - request_id
    - service and method name
    - duration
    - status (OK/ERROR)
    """

    async def intercept_service(
        self,
        continuation: Callable,
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """
        Intercept and log requests.
        
        Args:
            continuation: Next handler in chain
            handler_call_details: Request details
            
        Returns:
            RPC method handler
        """
        method_name = handler_call_details.method
        start_time = time.time()
        
        # Get request_id from metadata if available
        metadata_dict = dict(handler_call_details.invocation_metadata)
        request_id = metadata_dict.get("x-request-id", "unknown")
        
        logger.info(
            f"gRPC call started: method={method_name} request_id={request_id}"
        )
        
        try:
            handler = await continuation(handler_call_details)
            duration_ms = (time.time() - start_time) * 1000
            
            logger.info(
                f"gRPC call completed: method={method_name} "
                f"request_id={request_id} duration_ms={duration_ms:.2f} status=OK"
            )
            
            return handler
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            logger.error(
                f"gRPC call failed: method={method_name} "
                f"request_id={request_id} duration_ms={duration_ms:.2f} "
                f"status=ERROR error={str(e)}"
            )
            raise


class RateLimitInterceptor(ServerInterceptor):
    """
    Rate limiting interceptor for gRPC.
    
    Implements per-API-key rate limiting:
    - RPS (requests per second) limit
    - Concurrent requests limit
    
    Returns RESOURCE_EXHAUSTED on limit exceeded.
    """

    def __init__(
        self,
        rps_limit: int = 1000,
        concurrent_limit: int = 100,
    ):
        """
        Initialize rate limiter.
        
        Args:
            rps_limit: Max requests per second per API key (default: 1000)
            concurrent_limit: Max concurrent requests per API key (default: 100)
        """
        self.rps_limit = rps_limit
        self.concurrent_limit = concurrent_limit
        
        # State tracking
        self._request_counts: dict[str, list[float]] = defaultdict(list)
        self._concurrent_counts: dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()

    async def _check_rate_limit(self, api_key: str) -> tuple[bool, str]:
        """
        Check if request is within rate limits.
        
        Args:
            api_key: API key to check
            
        Returns:
            (allowed, reason) tuple
        """
        async with self._lock:
            now = time.time()
            
            # Clean old request timestamps (older than 1 second)
            self._request_counts[api_key] = [
                ts for ts in self._request_counts[api_key]
                if now - ts < 1.0
            ]
            
            # Check RPS limit
            if len(self._request_counts[api_key]) >= self.rps_limit:
                return False, f"RPS limit exceeded: {self.rps_limit}"
            
            # Check concurrent limit
            if self._concurrent_counts[api_key] >= self.concurrent_limit:
                return False, f"Concurrent limit exceeded: {self.concurrent_limit}"
            
            # Record request
            self._request_counts[api_key].append(now)
            self._concurrent_counts[api_key] += 1
            
            return True, ""

    async def _release_concurrent(self, api_key: str) -> None:
        """Release a concurrent request slot."""
        async with self._lock:
            self._concurrent_counts[api_key] = max(
                0, self._concurrent_counts[api_key] - 1
            )

    async def intercept_service(
        self,
        continuation: Callable,
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """
        Intercept and rate limit requests.
        
        Args:
            continuation: Next handler in chain
            handler_call_details: Request details
            
        Returns:
            RPC method handler
        """
        # Extract API key from metadata
        metadata_dict = dict(handler_call_details.invocation_metadata)
        api_key = metadata_dict.get("x-api-key", "anonymous")
        
        # Check rate limit
        allowed, reason = await self._check_rate_limit(api_key)
        
        if not allowed:
            logger.warning(
                f"Rate limit exceeded for {api_key}: {reason} "
                f"method={handler_call_details.method}"
            )
            
            async def abort_rate_limit(request, context):
                await context.abort(
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    reason,
                )
            
            return grpc.unary_unary_rpc_method_handler(abort_rate_limit)
        
        # Proceed with request
        try:
            handler = await continuation(handler_call_details)
            return handler
        finally:
            # Release concurrent slot
            await self._release_concurrent(api_key)
