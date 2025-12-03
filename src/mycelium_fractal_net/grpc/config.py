"""
Configuration for gRPC server.

Environment variables:
    GRPC_PORT: Server port (default: 50051)
    GRPC_MAX_WORKERS: Thread pool size (default: 10)
    GRPC_MAX_MESSAGE_SIZE: Max message size in MB (default: 10)
    GRPC_MAX_CONCURRENT_STREAMS: Max concurrent streams (default: 100)
    GRPC_KEEPALIVE_TIME_MS: Keepalive time (default: 30000)
    GRPC_KEEPALIVE_TIMEOUT_MS: Keepalive timeout (default: 10000)
    GRPC_TLS_ENABLED: Enable TLS (default: false)
    GRPC_TLS_CERT_PATH: Path to TLS certificate
    GRPC_TLS_KEY_PATH: Path to TLS private key
    GRPC_AUTH_ENABLED: Enable authentication (default: true)
    GRPC_RATE_LIMIT_RPS: Rate limit per API key (default: 1000)
    GRPC_RATE_LIMIT_CONCURRENT: Max concurrent requests per key (default: 50)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class GRPCConfig:
    """Configuration for gRPC server."""
    
    port: int
    max_workers: int
    max_message_size_mb: int
    max_concurrent_streams: int
    keepalive_time_ms: int
    keepalive_timeout_ms: int
    tls_enabled: bool
    tls_cert_path: Optional[str]
    tls_key_path: Optional[str]
    auth_enabled: bool
    rate_limit_rps: int
    rate_limit_concurrent: int
    
    @property
    def max_message_size(self) -> int:
        """Get max message size in bytes."""
        return self.max_message_size_mb * 1024 * 1024


def get_grpc_config() -> GRPCConfig:
    """
    Load gRPC configuration from environment variables.
    
    Returns:
        GRPCConfig: Server configuration.
    """
    return GRPCConfig(
        port=int(os.getenv("GRPC_PORT", "50051")),
        max_workers=int(os.getenv("GRPC_MAX_WORKERS", "10")),
        max_message_size_mb=int(os.getenv("GRPC_MAX_MESSAGE_SIZE", "10")),
        max_concurrent_streams=int(os.getenv("GRPC_MAX_CONCURRENT_STREAMS", "100")),
        keepalive_time_ms=int(os.getenv("GRPC_KEEPALIVE_TIME_MS", "30000")),
        keepalive_timeout_ms=int(os.getenv("GRPC_KEEPALIVE_TIMEOUT_MS", "10000")),
        tls_enabled=os.getenv("GRPC_TLS_ENABLED", "false").lower() == "true",
        tls_cert_path=os.getenv("GRPC_TLS_CERT_PATH"),
        tls_key_path=os.getenv("GRPC_TLS_KEY_PATH"),
        auth_enabled=os.getenv("GRPC_AUTH_ENABLED", "true").lower() == "true",
        rate_limit_rps=int(os.getenv("GRPC_RATE_LIMIT_RPS", "1000")),
        rate_limit_concurrent=int(os.getenv("GRPC_RATE_LIMIT_CONCURRENT", "50")),
    )
