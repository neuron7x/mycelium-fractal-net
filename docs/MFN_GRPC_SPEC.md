# MFN gRPC API Specification

## Overview

The MyceliumFractalNet (MFN) gRPC API provides high-throughput, production-grade access to MFN's simulation, validation, and feature extraction capabilities. This document describes the complete gRPC layer implementation.

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                     MFN gRPC Layer                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Client     │  │    Server    │  │ Interceptors │      │
│  │     SDK      │  │  Servicers   │  │    (Auth,    │      │
│  │              │  │              │  │ Audit, Rate) │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                  │                   │            │
│         └──────────────────┴───────────────────┘            │
│                           │                                 │
│                    ┌──────▼──────┐                          │
│                    │  Proto Defs │                          │
│                    │  (mfn.proto) │                          │
│                    └──────┬──────┘                          │
│                           │                                 │
└───────────────────────────┼─────────────────────────────────┘
                            │
                    ┌───────▼────────┐
                    │   MFN Core     │
                    │ (simulation,   │
                    │  validation,   │
                    │  features)     │
                    └────────────────┘
```

### Integration with MFN Architecture

The gRPC layer sits alongside the existing REST API (FastAPI) and provides:

- **Same business logic**: Uses the same core functions (`simulate_mycelium_field`, `run_validation`, `compute_fractal_features`)
- **No duplication**: Pure adapter layer that bridges between gRPC proto types and Python types
- **Independent operation**: Can run on different port, independently scaled
- **Shared security model**: Uses same API key authentication, compatible with existing auth system

### Layer Boundaries

```
External Clients
       │
       ├─── REST API (FastAPI) ──┐
       │                          │
       └─── gRPC API ─────────────┤
                                  │
                        ┌─────────▼─────────┐
                        │ Integration Layer │
                        │ (Schemas, Context)│
                        └─────────┬─────────┘
                                  │
                        ┌─────────▼─────────┐
                        │    MFN Core       │
                        │  (Pure Functions) │
                        └───────────────────┘
```

## Protocol Buffer Definitions

### Services

#### 1. MFNFeaturesService

Provides feature extraction operations.

**RPCs:**
- `ExtractFeatures`: Unary RPC - extract features from a single simulation
- `StreamFeatures`: Server streaming RPC - stream features during simulation

#### 2. MFNSimulationService

Provides simulation operations.

**RPCs:**
- `RunSimulation`: Unary RPC - run complete simulation
- `StreamSimulation`: Server streaming RPC - stream simulation state in real-time

#### 3. MFNValidationService

Provides validation operations.

**RPCs:**
- `ValidatePattern`: Unary RPC - validate pattern configuration

### Message Types

All request messages include:
- `request_id` (string): Unique request identifier

All response messages include:
- `request_id` (string): Echo of request ID
- `meta` (ResponseMeta): Metadata map with status and other info

See `grpc/proto/mfn.proto` for complete definitions.

## Server

### Starting the Server

**Programmatic:**
```python
import asyncio
from mycelium_fractal_net.grpc import serve_forever

asyncio.run(serve_forever())
```

**Command Line:**
```bash
python -m mycelium_fractal_net.grpc.server
```

### Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MFN_GRPC_PORT` | Server port | 50051 |
| `MFN_GRPC_MAX_WORKERS` | Thread pool size | 10 |
| `MFN_GRPC_MAX_MESSAGE_SIZE` | Max message size (MB) | 100 |
| `MFN_GRPC_MAX_CONCURRENT_STREAMS` | Max concurrent streams | 1000 |
| `MFN_GRPC_KEEPALIVE_TIME_MS` | Keepalive interval | 60000 |
| `MFN_GRPC_KEEPALIVE_TIMEOUT_MS` | Keepalive timeout | 20000 |
| `MFN_GRPC_TLS_ENABLED` | Enable TLS | false |
| `MFN_GRPC_TLS_CERT_PATH` | TLS certificate path | - |
| `MFN_GRPC_TLS_KEY_PATH` | TLS private key path | - |
| `MFN_API_KEYS` | Comma-separated API keys | - |

### Performance Settings

The server is configured for high throughput:

- **Target**: 40-60k RPS for unary RPCs
- **Streaming**: 100+ concurrent streams supported
- **Message size**: Up to 100MB per message
- **Keepalive**: Maintains long-lived connections

### TLS Configuration

Enable TLS for production:

```bash
export MFN_GRPC_TLS_ENABLED=true
export MFN_GRPC_TLS_CERT_PATH=/path/to/server.crt
export MFN_GRPC_TLS_KEY_PATH=/path/to/server.key
```

## Client SDK

### Installation

```bash
pip install mycelium-fractal-net
```

### Basic Usage

```python
import asyncio
from mycelium_fractal_net.grpc import MFNClient

async def main():
    async with MFNClient("localhost:50051", api_key="your-key") as client:
        # Run simulation
        result = await client.run_simulation(
            seed=42,
            grid_size=64,
            steps=64,
        )
        print(f"Fractal dimension: {result.fractal_dimension}")
        print(f"Growth events: {result.growth_events}")

asyncio.run(main())
```

### Streaming Example

```python
async def stream_simulation():
    async with MFNClient("localhost:50051") as client:
        async for frame in client.stream_simulation(
            seed=42,
            grid_size=64,
            steps=100,
            stream_interval=10,
        ):
            print(f"Step {frame.step}: {frame.fractal_dimension:.4f}")
```

### Client Configuration

```python
client = MFNClient(
    address="localhost:50051",
    api_key="your-api-key",           # Optional
    use_tls=True,                     # Enable TLS
    max_retries=3,                    # Retry transient errors
    retry_backoff_sec=1.0,            # Initial backoff
    timeout_sec=30.0,                 # RPC timeout
)
```

### Client Methods

#### Simulation

```python
result = await client.run_simulation(
    seed=42,
    grid_size=64,
    steps=64,
    alpha=0.18,
    spike_probability=0.25,
    turing_enabled=True,
    request_id=None,  # Auto-generated if None
)
```

#### Feature Extraction

```python
result = await client.extract_features(
    seed=42,
    grid_size=64,
    steps=64,
    alpha=0.18,
    spike_probability=0.25,
    turing_enabled=True,
)
```

#### Validation

```python
result = await client.validate_pattern(
    seed=42,
    epochs=1,
    batch_size=4,
    grid_size=64,
    steps=64,
    turing_enabled=True,
    quantum_jitter=False,
)
```

#### Streaming

```python
# Stream simulation
async for frame in client.stream_simulation(
    seed=42,
    grid_size=64,
    steps=100,
    stream_interval=10,
):
    # Process frame
    pass

# Stream features
async for frame in client.stream_features(
    seed=42,
    grid_size=64,
    steps=100,
    stream_interval=10,
):
    # Process frame
    pass
```

## Security

### Authentication

The gRPC API uses API key authentication with HMAC-SHA256 signatures.

#### Metadata Required

Clients must include these metadata fields:

- `x-api-key`: API key
- `x-signature`: HMAC-SHA256 signature
- `x-timestamp`: Unix timestamp

#### Signature Generation

```python
import hmac
import hashlib
import time

api_key = "your-api-key"
timestamp = str(time.time())
method = "/mfn.MFNSimulationService/RunSimulation"

# Compute signature
message = f"{timestamp}{method}".encode("utf-8")
signature = hmac.new(
    api_key.encode("utf-8"),
    message,
    hashlib.sha256,
).hexdigest()
```

The MFN client SDK handles this automatically.

#### Timestamp Validation

- Timestamps must be within 300 seconds (5 minutes) of server time
- Prevents replay attacks
- Tolerates reasonable clock skew

### Audit Logging

The audit interceptor logs all RPC calls with:

- `request_id`: Request identifier
- `method`: gRPC method name
- `duration_ms`: Request duration
- `status`: OK or ERROR

Logs are written to the standard Python logging system.

### Rate Limiting

The rate limiter enforces per-API-key limits:

- **RPS limit**: Max requests per second (default: 1000)
- **Concurrent limit**: Max concurrent requests (default: 100)

Returns `RESOURCE_EXHAUSTED` when limits are exceeded.

## Error Handling

### gRPC Status Codes

| Code | Description | When Used |
|------|-------------|-----------|
| `OK` | Success | Request completed successfully |
| `INVALID_ARGUMENT` | Bad request | Invalid parameters |
| `UNAUTHENTICATED` | Auth failed | Invalid API key or signature |
| `PERMISSION_DENIED` | Forbidden | Insufficient permissions |
| `RESOURCE_EXHAUSTED` | Rate limited | Too many requests |
| `INTERNAL` | Server error | Internal server error |
| `UNAVAILABLE` | Server down | Service unavailable |

### Error Details

All errors include:
- `request_id`: For tracking
- Descriptive message with debugging hints

### Client Retry Logic

The client automatically retries on:
- `UNAVAILABLE`
- `DEADLINE_EXCEEDED`
- `RESOURCE_EXHAUSTED`

Uses exponential backoff (1s, 2s, 4s, ...)

## Testing

### Running Tests

```bash
# All gRPC tests
make test-grpc

# Or with pytest
python -m pytest tests/grpc/ -v

# Specific test file
python -m pytest tests/grpc/test_grpc_server.py -v
```

### Test Coverage

- **Unit tests**: Server servicers, client SDK, interceptors
- **Integration tests**: Full client-server interaction
- **Streaming tests**: Bidirectional streaming
- **Auth tests**: Authentication and authorization
- **Rate limit tests**: Rate limiting behavior

## Development

### Generating Proto Stubs

```bash
# Generate Python stubs
make proto-gen

# Clean generated files
make proto-clean
```

### Manual Generation

```bash
python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --grpc_python_out=. \
    grpc/proto/mfn.proto
```

## CLI Usage Examples

### Using grpcurl

```bash
# List services
grpcurl -plaintext localhost:50051 list

# List methods
grpcurl -plaintext localhost:50051 list mfn.MFNSimulationService

# Call RunSimulation
grpcurl -plaintext -d '{
  "request_id": "test-123",
  "seed": 42,
  "grid_size": 64,
  "steps": 64
}' localhost:50051 mfn.MFNSimulationService/RunSimulation
```

### Using Python Client

```python
# Simple simulation
import asyncio
from mycelium_fractal_net.grpc import MFNClient

async def run():
    async with MFNClient("localhost:50051") as client:
        result = await client.run_simulation(seed=42, grid_size=64, steps=64)
        print(result)

asyncio.run(run())
```

## Production Deployment

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .
RUN pip install -e .

ENV MFN_GRPC_PORT=50051
ENV MFN_GRPC_TLS_ENABLED=true

CMD ["python", "-m", "mycelium_fractal_net.grpc.server"]
```

### Kubernetes

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mfn-grpc
spec:
  ports:
  - port: 50051
    protocol: TCP
  selector:
    app: mfn-grpc
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
        image: mfn:latest
        ports:
        - containerPort: 50051
        env:
        - name: MFN_GRPC_PORT
          value: "50051"
        - name: MFN_API_KEYS
          valueFrom:
            secretKeyRef:
              name: mfn-secrets
              key: api-keys
```

### Load Balancing

For horizontal scaling:
- Use gRPC load balancer (e.g., Envoy, NGINX)
- Enable keepalive to maintain persistent connections
- Monitor connection distribution across replicas

## Monitoring

### Metrics

Key metrics to monitor:

- **RPS**: Requests per second
- **Latency**: P50, P90, P99 latency
- **Error rate**: Failed requests percentage
- **Active streams**: Concurrent streaming RPCs
- **Connection count**: Active gRPC connections

### Integration with Prometheus

The server integrates with MFN's existing Prometheus metrics endpoint.

## Comparison with REST API

| Feature | gRPC | REST |
|---------|------|------|
| Protocol | HTTP/2 | HTTP/1.1 |
| Format | Protobuf | JSON |
| Streaming | Native | WebSocket |
| Performance | ~2-3x faster | Baseline |
| Type safety | Strong | Runtime |
| Browser support | Limited | Full |
| Use case | High-throughput integrations | Web apps, general API |

## FAQ

**Q: When should I use gRPC vs REST?**
A: Use gRPC for high-throughput backend integrations, microservices communication, and streaming. Use REST for web apps, mobile apps, and general-purpose API access.

**Q: Is gRPC backward compatible?**
A: Yes, protobuf ensures backward/forward compatibility when adding optional fields.

**Q: Can I use gRPC from browser?**
A: Yes, with grpc-web proxy. Native browser support is limited.

**Q: How do I debug gRPC calls?**
A: Use grpcurl CLI tool, or enable gRPC debug logging.

**Q: What's the max message size?**
A: Default 100MB, configurable via `MFN_GRPC_MAX_MESSAGE_SIZE`.

## References

- [gRPC Documentation](https://grpc.io/docs/)
- [Protocol Buffers Guide](https://protobuf.dev/)
- [MFN Architecture](ARCHITECTURE.md)
- [MFN Security](MFN_SECURITY.md)
