#!/usr/bin/env python
"""
Standalone gRPC server for MyceliumFractalNet.

Usage:
    python grpc_server.py [--port PORT] [--no-auth]

Environment Variables:
    GRPC_PORT - Server port (default: 50051)
    GRPC_MAX_WORKERS - Thread pool size (default: 10)
    GRPC_AUTH_ENABLED - Enable authentication (default: true)
    MFN_API_KEY - API key for authentication
"""

import asyncio
import sys
from mycelium_fractal_net.grpc.server import serve_forever

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(
        description="Start MFN gRPC server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("GRPC_PORT", "50051")),
        help="Server port (default: 50051)"
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable authentication (development only)"
    )
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["GRPC_PORT"] = str(args.port)
    if args.no_auth:
        os.environ["GRPC_AUTH_ENABLED"] = "false"
    
    print(f"Starting MFN gRPC server on port {args.port}...")
    print(f"Authentication: {'disabled' if args.no_auth else 'enabled'}")
    
    try:
        asyncio.run(serve_forever())
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)
