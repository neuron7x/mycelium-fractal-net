#!/usr/bin/env python3
"""
gRPC server launcher for MyceliumFractalNet v4.1.

This is a thin wrapper around the actual server implementation
in src/mycelium_fractal_net/grpc/server.py

Usage:
    python grpc_server.py --port 50051
"""

if __name__ == "__main__":
    from src.mycelium_fractal_net.grpc.server import main
    main()
