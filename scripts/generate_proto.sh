#!/bin/bash
# Generate Python code from protobuf definitions
# Usage: ./scripts/generate_proto.sh

set -e

echo "Generating Python code from protobuf definitions..."

# Ensure grpc_tools is installed
python -m pip install -q grpcio-tools

# Generate Python protobuf and gRPC code
python -m grpc_tools.protoc \
    -I. \
    --python_out=src/mycelium_fractal_net/grpc \
    --grpc_python_out=src/mycelium_fractal_net/grpc \
    protos/mycelium.proto

echo "✓ Generated protobuf code in src/mycelium_fractal_net/grpc/protos/"
echo "  - mycelium_pb2.py (protobuf messages)"
echo "  - mycelium_pb2_grpc.py (gRPC service stubs)"
