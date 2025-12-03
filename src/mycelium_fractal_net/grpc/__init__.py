"""
gRPC layer for MyceliumFractalNet.

Provides high-throughput gRPC API for simulation, validation, and feature extraction.

Components:
- server: Async gRPC server with servicers
- client: High-level client SDK with retry and authentication
- interceptors: Auth, audit, and rate limiting interceptors

Usage (Server):
    from mycelium_fractal_net.grpc import serve_forever
    import asyncio
    
    asyncio.run(serve_forever())

Usage (Client):
    from mycelium_fractal_net.grpc import MFNClient
    import asyncio
    
    async def main():
        async with MFNClient("localhost:50051", api_key="your-key") as client:
            result = await client.run_simulation(seed=42, grid_size=64, steps=64)
            print(f"Fractal dimension: {result.fractal_dimension}")
    
    asyncio.run(main())

Reference: docs/MFN_GRPC_SPEC.md
"""

from .client import MFNClient
from .interceptors import AuditInterceptor, AuthInterceptor, RateLimitInterceptor
from .server import serve, serve_forever

__all__ = [
    "MFNClient",
    "serve",
    "serve_forever",
    "AuthInterceptor",
    "AuditInterceptor",
    "RateLimitInterceptor",
]
