# MyceliumFractalNet gRPC API

**Version**: 4.1.0  
**Protocol**: gRPC (HTTP/2)  
**Port**: 50051 (default)

---

## Overview

The gRPC API provides high-performance, type-safe RPC interface for MyceliumFractalNet operations. It supports both unary (request-response) and streaming (real-time updates) RPCs.

### Benefits of gRPC

- **Performance**: Binary protocol (Protocol Buffers) with HTTP/2
- **Type Safety**: Strong typing with automatic client generation
- **Streaming**: Bidirectional streaming for real-time updates
- **Multi-language**: Auto-generated clients for 10+ languages
- **Efficiency**: Lower latency and bandwidth compared to REST/JSON

---

## Quick Start

### Start gRPC Server

```bash
# Basic server
python grpc_server.py --port 50051

# With authentication
export MFN_GRPC_AUTH_REQUIRED=true
export MFN_GRPC_API_KEY="your-secret-key"
python grpc_server.py --port 50051 --auth --api-key your-secret-key
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MFN_GRPC_PORT` | gRPC server port | 50051 |
| `MFN_GRPC_MAX_WORKERS` | Thread pool size | 10 |
| `MFN_GRPC_AUTH_REQUIRED` | Enable authentication | false |
| `MFN_GRPC_API_KEY` | API key for auth | "" |

---

## Service Definition

See `protos/mycelium.proto` for the complete service definition.

### Service Methods

| Method | Type | Description |
|--------|------|-------------|
| `HealthCheck` | Unary | Health check and uptime |
| `Validate` | Unary | Run validation cycle |
| `Simulate` | Unary | Simulate mycelium field |
| `SimulateStream` | Server Streaming | Real-time simulation updates |
| `ComputeNernst` | Unary | Calculate Nernst potential |
| `AggregateFederated` | Unary | Federated gradient aggregation |

---

## API Reference

### HealthCheck

Health check endpoint to verify server status.

**Request**: `HealthCheckRequest` (empty)

**Response**: `HealthCheckResponse`

```protobuf
message HealthCheckResponse {
  string status = 1;           // "healthy"
  string version = 2;          // "4.1.0"
  int64 uptime_seconds = 3;    // Server uptime
}
```

**Example** (Python):

```python
import grpc
from mycelium_pb2 import HealthCheckRequest
from mycelium_pb2_grpc import MyceliumServiceStub

channel = grpc.insecure_channel('localhost:50051')
stub = MyceliumServiceStub(channel)

response = stub.HealthCheck(HealthCheckRequest())
print(f"Status: {response.status}, Version: {response.version}")
```

---

### Validate

Run validation cycle with specified parameters.

**Request**: `ValidateRequest`

```protobuf
message ValidateRequest {
  int32 seed = 1;          // Random seed
  int32 epochs = 2;        // Training epochs
  int32 grid_size = 3;     // Grid size (default: 32)
}
```

**Response**: `ValidateResponse`

```protobuf
message ValidateResponse {
  double loss_start = 1;
  double loss_final = 2;
  double loss_drop = 3;
  double pot_min_mV = 4;
  double pot_max_mV = 5;
  double lyapunov_exponent = 6;
  double nernst_symbolic_mV = 7;
}
```

**Example** (Python):

```python
request = ValidateRequest(seed=42, epochs=5, grid_size=64)
response = stub.Validate(request)
print(f"Loss drop: {response.loss_drop:.3f}")
```

---

### Simulate

Simulate mycelium field evolution.

**Request**: `SimulateRequest`

```protobuf
message SimulateRequest {
  int32 seed = 1;
  int32 grid_size = 2;
  int32 steps = 3;
  bool turing_enabled = 4;
  double alpha = 5;
  double gamma = 6;
}
```

**Response**: `SimulateResponse`

```protobuf
message SimulateResponse {
  repeated double field_mean = 1;     // Mean potential per step
  repeated double field_std = 2;      // Std dev per step
  int32 growth_events = 3;            // Growth event count
  double fractal_dimension = 4;       // Box-counting dimension
  FieldStats field_stats = 5;         // Final field statistics
}
```

**Example** (Python):

```python
request = SimulateRequest(
    seed=42,
    grid_size=64,
    steps=100,
    turing_enabled=True,
    alpha=0.1,
    gamma=0.8
)
response = stub.Simulate(request)
print(f"Fractal dimension: {response.fractal_dimension:.3f}")
print(f"Growth events: {response.growth_events}")
```

---

### SimulateStream

Stream real-time simulation updates as the simulation progresses.

**Request**: `SimulateRequest` (same as Simulate)

**Response**: Stream of `SimulationUpdate`

```protobuf
message SimulationUpdate {
  int32 step = 1;              // Current step
  int32 total_steps = 2;       // Total steps
  double pot_mean_mV = 3;      // Mean potential (mV)
  double pot_std_mV = 4;       // Std dev (mV)
  int32 growth_events = 5;     // Growth events (final step only)
  bool completed = 6;          // True on last update
}
```

**Example** (Python):

```python
request = SimulateRequest(seed=42, grid_size=32, steps=50)

for update in stub.SimulateStream(request):
    print(f"Step {update.step}/{update.total_steps}: "
          f"pot_mean={update.pot_mean_mV:.2f} mV")
    
    if update.completed:
        print(f"Simulation complete! Growth events: {update.growth_events}")
        break
```

---

### ComputeNernst

Calculate Nernst potential for an ion.

**Request**: `NernstRequest`

```protobuf
message NernstRequest {
  int32 z_valence = 1;
  double concentration_out_molar = 2;
  double concentration_in_molar = 3;
  double temperature_k = 4;
}
```

**Response**: `NernstResponse`

```protobuf
message NernstResponse {
  double potential_mV = 1;
}
```

**Example** (Python):

```python
# K+ potential calculation
request = NernstRequest(
    z_valence=1,
    concentration_out_molar=0.005,  # 5 mM
    concentration_in_molar=0.140,   # 140 mM
    temperature_k=310.0             # 37°C
)
response = stub.ComputeNernst(request)
print(f"E_K = {response.potential_mV:.2f} mV")  # -89 mV
```

---

### AggregateFederated

Aggregate gradients from federated learning clients using Hierarchical Krum.

**Request**: `FederatedAggregateRequest`

```protobuf
message FederatedAggregateRequest {
  repeated Gradient gradients = 1;
  int32 num_clusters = 2;
  double byzantine_fraction = 3;
}

message Gradient {
  repeated double values = 1;
}
```

**Response**: `FederatedAggregateResponse`

```protobuf
message FederatedAggregateResponse {
  repeated double aggregated_gradient = 1;
  int32 num_accepted = 2;
  int32 num_rejected = 3;
}
```

**Example** (Python):

```python
gradients = [
    Gradient(values=[1.0, 2.0, 3.0]),
    Gradient(values=[1.1, 2.1, 2.9]),
    Gradient(values=[0.9, 1.9, 3.1]),
]

request = FederatedAggregateRequest(
    gradients=gradients,
    num_clusters=10,
    byzantine_fraction=0.2
)

response = stub.AggregateFederated(request)
print(f"Aggregated: {response.aggregated_gradient}")
print(f"Accepted: {response.num_accepted}, Rejected: {response.num_rejected}")
```

---

## Authentication

When authentication is enabled (`MFN_GRPC_AUTH_REQUIRED=true`), all RPC calls (except HealthCheck) require an API key in metadata.

### Adding API Key (Python)

```python
import grpc

# Create channel with API key metadata
def auth_interceptor(api_key):
    class AuthInterceptor(grpc.UnaryUnaryClientInterceptor):
        def intercept_unary_unary(self, continuation, client_call_details, request):
            metadata = []
            if client_call_details.metadata is not None:
                metadata = list(client_call_details.metadata)
            metadata.append(('x-api-key', api_key))
            
            new_details = client_call_details._replace(metadata=metadata)
            return continuation(new_details, request)
    
    return AuthInterceptor()

# Create authenticated channel
channel = grpc.insecure_channel('localhost:50051')
intercepted_channel = grpc.intercept_channel(
    channel,
    auth_interceptor('your-api-key')
)
stub = MyceliumServiceStub(intercepted_channel)
```

---

## Error Handling

gRPC errors are returned as `grpc.RpcError` with status codes:

| Status Code | Description |
|-------------|-------------|
| `OK` | Success |
| `INVALID_ARGUMENT` | Invalid request parameters |
| `UNAUTHENTICATED` | Missing or invalid API key |
| `INTERNAL` | Server-side error |
| `UNAVAILABLE` | Server not available |

**Example**:

```python
try:
    response = stub.Validate(request)
except grpc.RpcError as e:
    print(f"Error: {e.code()}, {e.details()}")
```

---

## Performance Tips

1. **Connection Reuse**: Reuse gRPC channels instead of creating new ones
2. **Streaming**: Use `SimulateStream` for large simulations to reduce memory
3. **Compression**: Enable gRPC compression for large payloads
4. **Parallelism**: Make concurrent RPCs with multiple threads/async

**Example** (Parallel Requests):

```python
import concurrent.futures

def run_simulation(seed):
    request = SimulateRequest(seed=seed, grid_size=32, steps=50)
    return stub.Simulate(request)

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(run_simulation, seed) for seed in range(100)]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]
```

---

## Client Code Generation

Generate gRPC clients for your language:

### Python

```bash
python -m grpc_tools.protoc -I. \
  --python_out=. \
  --grpc_python_out=. \
  protos/mycelium.proto
```

### Go

```bash
protoc -I. --go_out=. --go-grpc_out=. protos/mycelium.proto
```

### Node.js

```bash
npm install @grpc/grpc-js @grpc/proto-loader
node generate_client.js  # Use proto-loader at runtime
```

### Java

```bash
protoc -I. --java_out=. --grpc-java_out=. protos/mycelium.proto
```

---

## Deployment

### Docker

```dockerfile
# Add gRPC server to Dockerfile
EXPOSE 50051
CMD ["python", "grpc_server.py", "--port", "50051"]
```

### Kubernetes

Update service to expose gRPC port (already added to `k8s.yaml`):

```yaml
ports:
  - port: 80
    targetPort: 8000
    name: http
  - port: 50051
    targetPort: 50051
    name: grpc
```

---

## Monitoring

gRPC server metrics can be monitored via:

1. **Application logs**: Structured JSON logs with request/response details
2. **Prometheus metrics**: Expose gRPC metrics (requires grpc-prometheus)
3. **OpenTelemetry**: Distributed tracing for gRPC calls

---

## References

- [gRPC Official Docs](https://grpc.io/docs/)
- [Protocol Buffers Guide](https://protobuf.dev/)
- [MFN Integration Spec](MFN_INTEGRATION_SPEC.md)
- [MFN System Role](MFN_SYSTEM_ROLE.md)
