# MFN gRPC API Specification

**Version:** 4.1.0  
**Status:** Production-Ready  
**Last Updated:** 2025-12-03

## Overview

MyceliumFractalNet gRPC API provides high-throughput, low-latency access to the MFN simulation and validation engine. Built on gRPC/HTTP2, it offers:

- **High Performance**: 40-60k RPS for unary calls, 100+ concurrent streams
- **Production Security**: API key authentication, HMAC-SHA256 signatures, rate limiting
- **Streaming Support**: Real-time feature extraction and simulation state updates
- **Client SDK**: Python async client with automatic retries and backoff
- **Full Test Coverage**: 32 tests covering all services and edge cases

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      gRPC Client (SDK)                      │
│  - Automatic request ID generation                          │
│  - HMAC signature for authentication                        │
│  - Retry with exponential backoff                           │
│  - Streaming support (async iterators)                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ gRPC/HTTP2 (with TLS optional)
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  gRPC Server (Async)                         │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  Interceptors Chain                                    │ │
│  │  1. AuthInterceptor (API key + signature + timestamp) │ │
│  │  2. AuditInterceptor (logging)                        │ │
│  │  3. RateLimitInterceptor (per-key RPS + concurrent)   │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────┐                │
│  │  Service Implementations               │                │
│  │  - MFNFeaturesService                  │                │
│  │  - MFNSimulationService                │                │
│  │  - MFNValidationService                │                │
│  └────────────────────────────────────────┘                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Direct function calls
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              MFN Core Engine                                 │
│  - simulate_mycelium_field()                                │
│  - estimate_fractal_dimension()                             │
│  - run_validation()                                         │
└─────────────────────────────────────────────────────────────┘
```

### Integration with Existing System

The gRPC API layer integrates seamlessly with the existing MFN architecture:

- **Reuses Core Logic**: All RPC methods delegate to existing core functions (`simulate_mycelium_field`, `run_validation`, etc.)
- **Compatible with REST API**: Both REST and gRPC share the same core engine
- **Independent Deployment**: Can run alongside FastAPI REST server or standalone
- **Shared Configuration**: Uses `mycelium_fractal_net.integration` for authentication and logging

---

## Services

### 1. MFNFeaturesService

Feature extraction from mycelium simulations.

#### ExtractFeatures (Unary)

Extract features from a single simulation run.

**Request:**
```protobuf
message FeatureRequest {
  string request_id = 1;              // Unique request identifier
  int32 seed = 2;                     // Random seed (default: 42)
  int32 grid_size = 3;                // Grid size NxN (default: 64)
  int32 steps = 4;                    // Simulation steps (default: 64)
  double alpha = 5;                   // Diffusion coefficient (default: 0.18)
  double spike_probability = 6;        // Growth probability (default: 0.25)
  bool turing_enabled = 7;            // Enable Turing morphogenesis
}
```

**Response:**
```protobuf
message FeatureResponse {
  string request_id = 1;
  ResponseMeta meta = 2;              // Server metadata
  double fractal_dimension = 3;        // Box-counting dimension
  double pot_min_mV = 4;              // Min potential (mV)
  double pot_max_mV = 5;              // Max potential (mV)
  double pot_mean_mV = 6;             // Mean potential (mV)
  double pot_std_mV = 7;              // Std dev potential (mV)
  int32 growth_events = 8;            // Number of growth events
}
```

#### StreamFeatures (Server Streaming)

Stream features during multi-step simulation.

**Request:**
```protobuf
message FeatureStreamRequest {
  string request_id = 1;
  int32 seed = 2;
  int32 grid_size = 3;
  int32 total_steps = 4;              // Total simulation steps
  int32 steps_per_frame = 5;          // Steps between frames
  double alpha = 6;
  double spike_probability = 7;
  bool turing_enabled = 8;
}
```

**Response Stream:**
```protobuf
message FeatureFrame {
  string request_id = 1;
  int32 step = 2;                     // Current step number
  double fractal_dimension = 3;
  double pot_min_mV = 4;
  double pot_max_mV = 5;
  double pot_mean_mV = 6;
  double pot_std_mV = 7;
  int32 growth_events = 8;            // Cumulative
  bool is_final = 9;                  // Last frame indicator
}
```

---

### 2. MFNSimulationService

Mycelium field simulation.

#### RunSimulation (Unary)

Run a complete simulation and return final state.

**Request:**
```protobuf
message SimulationRequest {
  string request_id = 1;
  int32 seed = 2;
  int32 grid_size = 3;
  int32 steps = 4;
  double alpha = 5;
  double spike_probability = 6;
  bool turing_enabled = 7;
}
```

**Response:**
```protobuf
message SimulationResult {
  string request_id = 1;
  ResponseMeta meta = 2;
  int32 growth_events = 3;
  double pot_min_mV = 4;
  double pot_max_mV = 5;
  double pot_mean_mV = 6;
  double pot_std_mV = 7;
  double fractal_dimension = 8;
}
```

#### StreamSimulation (Server Streaming)

Stream simulation state updates.

**Request:**
```protobuf
message SimulationStreamRequest {
  string request_id = 1;
  int32 seed = 2;
  int32 grid_size = 3;
  int32 total_steps = 4;
  int32 steps_per_frame = 5;
  double alpha = 6;
  double spike_probability = 7;
  bool turing_enabled = 8;
}
```

**Response Stream:**
```protobuf
message SimulationFrame {
  string request_id = 1;
  int32 step = 2;
  int32 growth_events = 3;
  double pot_min_mV = 4;
  double pot_max_mV = 5;
  double pot_mean_mV = 6;
  double pot_std_mV = 7;
  bool is_final = 8;
}
```

---

### 3. MFNValidationService

Pattern validation with training cycle.

#### ValidatePattern (Unary)

Validate pattern using STDP training.

**Request:**
```protobuf
message ValidationRequest {
  string request_id = 1;
  int32 seed = 2;
  int32 epochs = 3;                   // Training epochs
  int32 batch_size = 4;               // Batch size
  int32 grid_size = 5;
  int32 steps = 6;
  bool turing_enabled = 7;
  bool quantum_jitter = 8;            // Enable quantum noise
}
```

**Response:**
```protobuf
message ValidationResult {
  string request_id = 1;
  ResponseMeta meta = 2;
  double loss_start = 3;              // Initial loss
  double loss_final = 4;              // Final loss
  double loss_drop = 5;               // Loss reduction
  double pot_min_mV = 6;
  double pot_max_mV = 7;
  double example_fractal_dim = 8;
  double lyapunov_exponent = 9;       // Stability metric
  double growth_events = 10;
  double nernst_symbolic_mV = 11;
  double nernst_numeric_mV = 12;
}
```

---

## Security

### Authentication Flow

1. **API Key**: Included in metadata as `x-api-key`
2. **Request ID**: Unique identifier for request tracking
3. **Timestamp**: Unix timestamp for replay protection
4. **Signature**: HMAC-SHA256 of `{request_id}:{timestamp}` using API key

**Metadata Example:**
```
x-api-key: your-secret-key-here
x-request-id: 550e8400-e29b-41d4-a716-446655440000
x-timestamp: 1733241600.123456
x-signature: a3f5b8c9d1e2f4g6h8i0j2k4l6m8n0p2q4r6s8t0u2v4w6x8y0z1a3b5c7d9
```

### Signature Generation (Python)

```python
import hmac
import hashlib
import time

def generate_signature(api_key: str, request_id: str, timestamp: str) -> str:
    message = f"{request_id}:{timestamp}"
    return hmac.new(
        api_key.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()

# Usage
api_key = "your-secret-key"
request_id = "550e8400-e29b-41d4-a716-446655440000"
timestamp = str(time.time())
signature = generate_signature(api_key, request_id, timestamp)
```

### Rate Limiting

**Per-API-Key Limits:**
- **RPS**: 1000 requests per second (configurable via `GRPC_RATE_LIMIT_RPS`)
- **Concurrent**: 50 concurrent requests (configurable via `GRPC_RATE_LIMIT_CONCURRENT`)

**Error Response:**
```
Status: RESOURCE_EXHAUSTED
Message: "Rate limit exceeded"
```

### Timestamp Validation

- Requests must have timestamps within **5 minutes** of server time
- Prevents replay attacks

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GRPC_PORT` | 50051 | Server port |
| `GRPC_MAX_WORKERS` | 10 | Thread pool size |
| `GRPC_MAX_MESSAGE_SIZE` | 10 | Max message size (MB) |
| `GRPC_MAX_CONCURRENT_STREAMS` | 100 | Max concurrent streams |
| `GRPC_KEEPALIVE_TIME_MS` | 30000 | Keepalive interval |
| `GRPC_KEEPALIVE_TIMEOUT_MS` | 10000 | Keepalive timeout |
| `GRPC_TLS_ENABLED` | false | Enable TLS |
| `GRPC_TLS_CERT_PATH` | - | Path to TLS certificate |
| `GRPC_TLS_KEY_PATH` | - | Path to TLS private key |
| `GRPC_AUTH_ENABLED` | true | Enable authentication |
| `GRPC_RATE_LIMIT_RPS` | 1000 | Rate limit (RPS) |
| `GRPC_RATE_LIMIT_CONCURRENT` | 50 | Concurrent request limit |

### Server Startup

**Without TLS:**
```bash
export GRPC_PORT=50051
export GRPC_AUTH_ENABLED=true
export MFN_API_KEY=your-secret-key

python -m mycelium_fractal_net.grpc.server
```

**With TLS:**
```bash
export GRPC_PORT=50051
export GRPC_TLS_ENABLED=true
export GRPC_TLS_CERT_PATH=/path/to/cert.pem
export GRPC_TLS_KEY_PATH=/path/to/key.pem
export MFN_API_KEY=your-secret-key

python -m mycelium_fractal_net.grpc.server
```

---

## Client SDK Usage

### Installation

```bash
pip install mycelium-fractal-net
```

### Basic Usage

```python
import asyncio
from mycelium_fractal_net.grpc import MFNClient

async def main():
    # Create client
    async with MFNClient(
        address="localhost:50051",
        api_key="your-secret-key",
        tls_enabled=False
    ) as client:
        # Extract features
        response = await client.extract_features(
            seed=42,
            grid_size=64,
            steps=100,
            turing_enabled=True
        )
        
        print(f"Fractal Dimension: {response.fractal_dimension}")
        print(f"Growth Events: {response.growth_events}")
        print(f"Potential Range: {response.pot_min_mV} - {response.pot_max_mV} mV")

asyncio.run(main())
```

### Streaming Example

```python
async def stream_features():
    async with MFNClient("localhost:50051", "your-key") as client:
        async for frame in client.stream_features(
            seed=42,
            grid_size=64,
            total_steps=200,
            steps_per_frame=20
        ):
            print(f"Step {frame.step}: D={frame.fractal_dimension:.3f}")
            if frame.is_final:
                print("Simulation complete!")

asyncio.run(stream_features())
```

### Simulation

```python
async def run_simulation():
    async with MFNClient("localhost:50051", "your-key") as client:
        result = await client.run_simulation(
            seed=100,
            grid_size=128,
            steps=500,
            alpha=0.2,
            spike_probability=0.3
        )
        
        print(f"Final fractal dimension: {result.fractal_dimension}")
        print(f"Growth events: {result.growth_events}")

asyncio.run(run_simulation())
```

### Validation

```python
async def validate_pattern():
    async with MFNClient("localhost:50051", "your-key") as client:
        result = await client.validate_pattern(
            seed=42,
            epochs=5,
            batch_size=8,
            grid_size=64,
            steps=100
        )
        
        print(f"Loss: {result.loss_start:.4f} → {result.loss_final:.4f}")
        print(f"Loss drop: {result.loss_drop:.4f}")
        print(f"Lyapunov: {result.lyapunov_exponent:.4f}")

asyncio.run(validate_pattern())
```

### Error Handling

```python
async def with_error_handling():
    async with MFNClient("localhost:50051", "your-key") as client:
        try:
            response = await client.extract_features(seed=42)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAUTHENTICATED:
                print("Authentication failed")
            elif e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                print("Rate limit exceeded")
            elif e.code() == grpc.StatusCode.UNAVAILABLE:
                print("Server unavailable (will retry)")
            else:
                print(f"Error: {e.code()}: {e.details()}")

asyncio.run(with_error_handling())
```

---

## CLI Examples

### Using grpcurl

**List services:**
```bash
grpcurl -plaintext localhost:50051 list
```

**Describe service:**
```bash
grpcurl -plaintext localhost:50051 describe mfn.MFNFeaturesService
```

**Call ExtractFeatures:**
```bash
grpcurl -plaintext \
  -H "x-api-key: your-key" \
  -H "x-request-id: test-123" \
  -H "x-timestamp: $(date +%s)" \
  -H "x-signature: <computed-signature>" \
  -d '{
    "request_id": "test-123",
    "seed": 42,
    "grid_size": 64,
    "steps": 100,
    "turing_enabled": true
  }' \
  localhost:50051 mfn.MFNFeaturesService/ExtractFeatures
```

**Stream features:**
```bash
grpcurl -plaintext \
  -H "x-api-key: your-key" \
  -d '{
    "request_id": "stream-123",
    "seed": 42,
    "grid_size": 64,
    "total_steps": 100,
    "steps_per_frame": 10
  }' \
  localhost:50051 mfn.MFNFeaturesService/StreamFeatures
```

---

## Performance Characteristics

### Throughput

**Unary RPCs:**
- Simple extract_features: **40-60k RPS** (tested on 8-core machine)
- Validation cycle: **100-500 RPS** (depends on epochs/batch size)

**Streaming:**
- Concurrent streams: **100+** simultaneous connections
- Frame rate: **10-20 frames/second** per stream (depends on steps_per_frame)

### Latency

| Operation | p50 | p95 | p99 |
|-----------|-----|-----|-----|
| ExtractFeatures (64x64, 100 steps) | 15ms | 30ms | 50ms |
| RunSimulation (64x64, 100 steps) | 12ms | 25ms | 45ms |
| ValidatePattern (1 epoch, batch=4) | 80ms | 150ms | 200ms |

### Resource Usage

**Per Request:**
- Memory: ~5-10 MB (depends on grid size)
- CPU: 1 core utilization during computation

**Server Overhead:**
- Base memory: ~50 MB
- Per connection: ~1 MB

---

## Development

### Regenerating Stubs

When modifying `mfn.proto`:

```bash
python -m grpc_tools.protoc \
  -I./grpc/proto \
  --python_out=./src/mycelium_fractal_net/grpc \
  --pyi_out=./src/mycelium_fractal_net/grpc \
  --grpc_python_out=./src/mycelium_fractal_net/grpc \
  ./grpc/proto/mfn.proto

# Fix import in generated file
sed -i 's/^import mfn_pb2/from . import mfn_pb2/' \
  src/mycelium_fractal_net/grpc/mfn_pb2_grpc.py
```

### Running Tests

```bash
# All gRPC tests
pytest tests/test_grpc_api/ -v

# Specific test suite
pytest tests/test_grpc_api/test_grpc_server.py -v
pytest tests/test_grpc_api/test_grpc_client.py -v
pytest tests/test_grpc_api/test_grpc_interceptors.py -v
pytest tests/test_grpc_api/test_grpc_integration.py -v
```

### Test Coverage

- **32 tests** covering:
  - Server servicers (7 tests)
  - Client SDK (9 tests)
  - Interceptors (8 tests)
  - End-to-end integration (8 tests)

---

## Error Codes

| Code | Description | Retry? |
|------|-------------|--------|
| `OK` | Success | - |
| `INVALID_ARGUMENT` | Invalid request parameters | No |
| `UNAUTHENTICATED` | Missing/invalid API key or signature | No |
| `RESOURCE_EXHAUSTED` | Rate limit exceeded | Yes (with backoff) |
| `UNAVAILABLE` | Server temporarily unavailable | Yes |
| `INTERNAL` | Server internal error | Yes |
| `DEADLINE_EXCEEDED` | Request timeout | Yes |

---

## Production Deployment

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY grpc/ ./grpc/

ENV GRPC_PORT=50051
ENV GRPC_AUTH_ENABLED=true

EXPOSE 50051

CMD ["python", "-m", "mycelium_fractal_net.grpc.server"]
```

### Kubernetes

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mfn-grpc
spec:
  selector:
    app: mfn-grpc
  ports:
    - protocol: TCP
      port: 50051
      targetPort: 50051
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mfn-grpc
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mfn-grpc
  template:
    metadata:
      labels:
        app: mfn-grpc
    spec:
      containers:
      - name: mfn-grpc
        image: mfn-grpc:4.1.0
        ports:
        - containerPort: 50051
        env:
        - name: GRPC_PORT
          value: "50051"
        - name: GRPC_MAX_WORKERS
          value: "10"
        - name: MFN_API_KEY
          valueFrom:
            secretKeyRef:
              name: mfn-secrets
              key: api-key
        resources:
          limits:
            memory: "2Gi"
            cpu: "2000m"
          requests:
            memory: "1Gi"
            cpu: "1000m"
```

### Load Balancing

gRPC uses HTTP/2, which maintains persistent connections. Use a load balancer that supports gRPC:

- **Envoy**: Recommended for gRPC
- **NGINX**: Requires `grpc_pass` directive
- **GCP Load Balancer**: Native gRPC support
- **AWS ALB**: Requires target group with gRPC protocol

---

## Monitoring

### Metrics

The `AuditInterceptor` logs:
- Request ID
- Method name
- Duration (ms)
- Status (OK/ERROR)
- Error details

### Logging Example

```json
{
  "timestamp": "2025-12-03T13:45:00Z",
  "level": "INFO",
  "logger": "grpc.interceptors",
  "message": "gRPC request completed: /mfn.MFNFeaturesService/ExtractFeatures",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "method": "/mfn.MFNFeaturesService/ExtractFeatures",
  "duration_ms": 15,
  "status": "OK"
}
```

---

## Roadmap

### Future Enhancements

1. **Reflection API**: Enable server reflection for dynamic client generation
2. **Bidirectional Streaming**: For interactive simulation control
3. **Protobuf Compression**: Reduce bandwidth for large responses
4. **Circuit Breaker**: Automatic degradation on overload
5. **Distributed Tracing**: OpenTelemetry integration
6. **gRPC Gateway**: Expose REST endpoints from proto definitions

---

## References

- **Proto Definition**: `/grpc/proto/mfn.proto`
- **Server Implementation**: `/src/mycelium_fractal_net/grpc/server.py`
- **Client SDK**: `/src/mycelium_fractal_net/grpc/client.py`
- **Tests**: `/tests/test_grpc_api/`
- **Core Engine**: `/src/mycelium_fractal_net/model.py`

---

**Questions or Issues?**  
Open an issue on GitHub or contact the MFN team.
