# MFN gRPC API

This directory contains the Protocol Buffer definitions for the MyceliumFractalNet gRPC API.

## Contents

- `proto/mfn.proto` - Service and message definitions

## Generating Stubs

To regenerate Python stubs after modifying the proto file:

```bash
cd /path/to/mycelium-fractal-net

python -m grpc_tools.protoc \
  -I./grpc/proto \
  --python_out=./src/mycelium_fractal_net/grpc \
  --pyi_out=./src/mycelium_fractal_net/grpc \
  --grpc_python_out=./src/mycelium_fractal_net/grpc \
  ./grpc/proto/mfn.proto

# Fix the import in the generated gRPC stub
sed -i 's/^import mfn_pb2/from . import mfn_pb2/' \
  src/mycelium_fractal_net/grpc/mfn_pb2_grpc.py
```

## Services

### MFNFeaturesService
- `ExtractFeatures` - Extract fractal features from simulation
- `StreamFeatures` - Stream features during simulation

### MFNSimulationService
- `RunSimulation` - Run complete simulation
- `StreamSimulation` - Stream simulation state updates

### MFNValidationService
- `ValidatePattern` - Validate pattern with training cycle

## Documentation

See [docs/MFN_GRPC_SPEC.md](../docs/MFN_GRPC_SPEC.md) for complete API documentation, usage examples, and deployment guides.

## Quick Start

### Server

```bash
export GRPC_PORT=50051
export MFN_API_KEY=your-secret-key
python -m mycelium_fractal_net.grpc.server
```

### Client

```python
from mycelium_fractal_net.grpc import MFNClient
import asyncio

async def main():
    async with MFNClient("localhost:50051", "your-secret-key") as client:
        response = await client.extract_features(seed=42, grid_size=64, steps=100)
        print(f"Fractal dimension: {response.fractal_dimension}")

asyncio.run(main())
```

## Testing

```bash
pytest tests/test_grpc_api/ -v
```

32 tests covering:
- Server servicers
- Client SDK
- Interceptors (auth, audit, rate-limit)
- End-to-end integration
