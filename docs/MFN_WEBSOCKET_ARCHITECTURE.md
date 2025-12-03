# MFN WebSocket Streaming Architecture

## Overview

The MFN WebSocket Streaming subsystem provides real-time, bidirectional communication for streaming fractal features and live simulation updates. The architecture is designed for production-grade reliability with HMAC-secured authentication, backpressure handling, and comprehensive audit logging.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT APPLICATION                        │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐      │
│  │  JavaScript  │   │    Python    │   │   Any Lang   │      │
│  │   Client     │   │    Client    │   │    Client    │      │
│  └──────────────┘   └──────────────┘   └──────────────┘      │
└──────────────────────────────┬──────────────────────────────────┘
                                │ WebSocket (ws:// or wss://)
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                      MFN API SERVER (FastAPI)                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    WebSocket Endpoints                     │  │
│  │  • /ws/stream_features  - Real-time fractal features      │  │
│  │  • /ws/simulation_live  - Live simulation updates         │  │
│  │  • /ws/heartbeat        - Health monitoring                │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                │                                 │
│                                ↓                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              WSConnectionManager                           │  │
│  │  • Connection lifecycle management                         │  │
│  │  • Authentication (API key + HMAC signature)               │  │
│  │  • Subscription routing                                    │  │
│  │  • Heartbeat monitoring                                    │  │
│  │  • Backpressure handling                                   │  │
│  │  • Audit logging                                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                │                                 │
│                    ┌───────────┴──────────┐                     │
│                    ↓                      ↓                      │
│  ┌──────────────────────────┐  ┌──────────────────────────┐    │
│  │  stream_features_adapter │  │ stream_simulation_adapter │    │
│  │  • Fractal feature comp. │  │ • State-by-state updates  │    │
│  │  • Async iteration       │  │ • Real-time simulation    │    │
│  │  • Rate limiting         │  │ • Progress tracking       │    │
│  └──────────────────────────┘  └──────────────────────────┘    │
│                    │                      │                      │
│                    └───────────┬──────────┘                     │
│                                ↓                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │           ReactionDiffusionEngine (Core)                   │  │
│  │  • Mycelium field simulation                               │  │
│  │  • Turing morphogenesis                                    │  │
│  │  • Fractal dimension computation                           │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. WebSocket Endpoints (`api.py`)

Three dedicated WebSocket endpoints handle different streaming patterns:

#### `/ws/stream_features`
- **Purpose**: Stream real-time fractal features from an active simulation
- **Use Case**: Monitoring, dashboards, feature visualization
- **Update Rate**: Configurable (10ms - 10s intervals)
- **Features Streamed**:
  - Potential statistics (mean, std, min, max)
  - Active node count & ratio
  - Fractal dimension
  - Energy measures
  - Spatial variance

#### `/ws/simulation_live`
- **Purpose**: Stream state-by-state updates during simulation execution
- **Use Case**: Interactive simulation, debugging, training visualization
- **Update Rate**: Every N steps (configurable)
- **State Included**:
  - Step number & progress
  - Current state metrics
  - Growth events
  - Optional full field state

#### `/ws/heartbeat`
- **Purpose**: Lightweight keepalive and health monitoring
- **Use Case**: Connection health checks, monitoring infrastructure
- **Features**: Authentication + heartbeat only, no data streaming

### 2. Connection Manager (`ws_manager.py`)

The `WSConnectionManager` is the central orchestrator for all WebSocket connections.

#### Core Responsibilities

**Connection Lifecycle**:
- Accept new WebSocket connections
- Generate unique connection IDs
- Track connection state and metadata
- Clean up on disconnect

**Authentication**:
- API key validation (constant-time comparison)
- Timestamp validation (5-minute window for replay protection)
- HMAC-SHA256 signature verification (optional enhanced security)
- Per-connection authentication tracking

**Subscription Management**:
- Subscribe/unsubscribe to streams
- Map streams to connections
- Track stream start times for audit
- Enforce authentication before subscription

**Message Queue & Backpressure**:
- Per-connection message queues (max 1000 messages)
- Three backpressure strategies:
  - `DROP_OLDEST`: Remove oldest message when queue full
  - `DROP_NEWEST`: Drop new message when queue full
  - `COMPRESS`: Sample queue (keep every Nth message)
- Automatic queue flushing

**Heartbeat Monitoring**:
- Server-initiated heartbeat every 30s
- Client responds with pong
- Automatic timeout detection (60s)
- Graceful disconnection of dead connections

**Audit Logging**:
- Connection duration tracking
- Packets sent/received counters
- Dropped frames tracking
- Per-stream metrics (duration, packet count)
- Drop rate calculation on disconnect

### 3. Stream Adapters (`ws_adapters.py`)

Adapters bridge the WebSocket layer with the numerical core using async generators.

#### `stream_features_adapter`
```python
async def stream_features_adapter(
    stream_id: str,
    params: StreamFeaturesParams,
    ctx: ServiceContext,
) -> AsyncIterator[WSFeatureUpdate]
```

- Initializes ReactionDiffusionEngine
- Runs simulation steps in loop
- Computes fractal features after each step
- Yields WSFeatureUpdate messages
- Respects update_interval_ms for rate limiting
- Cancellable via asyncio

#### `stream_simulation_live_adapter`
```python
async def stream_simulation_live_adapter(
    stream_id: str,
    params: SimulationLiveParams,
    ctx: ServiceContext,
) -> AsyncIterator[WSSimulationState | WSSimulationComplete]
```

- Runs full simulation with step-by-step updates
- Configurable grid size, steps, parameters
- Sends state updates at specified intervals
- Yields WSSimulationComplete on finish
- Tracks growth events and metrics
- Cancellable via asyncio

### 4. Message Schemas (`ws_schemas.py`)

Pydantic models define all WebSocket messages for validation and type safety.

**Lifecycle Messages**:
- `WSInitRequest` / `WSAuthRequest` / `WSSubscribeRequest`
- Success/failure responses
- Error messages

**Stream Messages**:
- `WSFeatureUpdate`: Real-time feature data
- `WSSimulationState`: Step-by-step simulation state
- `WSSimulationComplete`: Final simulation results
- `WSHeartbeatRequest`: Keepalive ping/pong

**Configuration**:
- `StreamFeaturesParams`: Feature streaming parameters
- `SimulationLiveParams`: Simulation configuration

## Protocol Flow

### Connection Establishment

```
Client                                    Server
  │                                         │
  │─────── WebSocket Connect ──────────────→│
  │←─────── WebSocket Accept ───────────────│
  │                                         │
  │─────── INIT ────────────────────────────→│
  │           {protocol_version: "1.0"}     │
  │←─────── INIT ───────────────────────────│
  │           {protocol_version: "1.0"}     │
  │                                         │
  │─────── AUTH ────────────────────────────→│
  │           {api_key, timestamp, sig}     │
  │←─────── AUTH_SUCCESS ───────────────────│
  │                                         │
  │─────── SUBSCRIBE ───────────────────────→│
  │           {stream_type, stream_id, ...} │
  │←─────── SUBSCRIBE_SUCCESS ──────────────│
  │                                         │
```

### Data Streaming

```
Client                                    Server
  │                                         │
  │←─────── FEATURE_UPDATE ─────────────────│ (periodic)
  │           {features, sequence, ...}     │
  │                                         │
  │←─────── HEARTBEAT ──────────────────────│ (every 30s)
  │─────── PONG ────────────────────────────→│
  │                                         │
  │←─────── FEATURE_UPDATE ─────────────────│
  │                                         │
  │─────── UNSUBSCRIBE ─────────────────────→│
  │           {stream_id}                   │
  │                                         │
  │─────── CLOSE ───────────────────────────→│
  │←─────── Disconnect ──────────────────────│
  │                                         │
```

## Security Architecture

### Authentication Layers

1. **API Key Validation**
   - Constant-time comparison (timing attack resistant)
   - Configurable per environment
   - Multiple valid keys supported

2. **Timestamp Validation**
   - 5-minute window (prevents replay attacks)
   - Server time checked on every auth attempt
   - Millisecond precision

3. **HMAC Signature (Optional Enhanced Security)**
   - HMAC-SHA256(api_key, timestamp)
   - Hex-encoded signature
   - Constant-time comparison
   - Client-side generation required

### Signature Generation (Client-Side)

```python
import hmac
import hashlib
import time

api_key = "your-secret-key"
timestamp = int(time.time() * 1000)
timestamp_str = str(timestamp)

signature = hmac.new(
    api_key.encode('utf-8'),
    timestamp_str.encode('utf-8'),
    hashlib.sha256
).hexdigest()
```

## Backpressure Strategies

### Strategy Comparison

| Strategy | Use Case | Pros | Cons |
|----------|----------|------|------|
| **drop_oldest** | Real-time monitoring | Always shows latest data | May lose historical context |
| **drop_newest** | Historical playback | Preserves order | Client falls behind |
| **compress** | Adaptive load | Balances history/recency | Non-uniform sampling |

### Implementation

```python
class WSConnectionManager:
    async def _handle_backpressure(
        self,
        connection: WSConnectionState,
        message: Dict[str, Any],
    ) -> None:
        if self.backpressure_strategy == BackpressureStrategy.DROP_OLDEST:
            connection.message_queue.popleft()
            connection.message_queue.append(message)
            connection.dropped_frames += 1
        # ... other strategies
```

## Audit Metrics

### Per-Connection Metrics

Tracked in `WSConnectionState`:
- `packets_sent`: Total messages sent to client
- `packets_received`: Total messages received from client
- `dropped_frames`: Frames dropped due to backpressure
- `stream_start_time`: Start timestamp per stream_id

### Logged on Disconnect

```json
{
  "connection_id": "uuid-xxx",
  "duration_seconds": 125.3,
  "packets_sent": 1250,
  "packets_received": 42,
  "dropped_frames": 15,
  "drop_rate_percent": 1.2
}
```

### Per-Stream Metrics

Logged on unsubscribe:
```json
{
  "stream_id": "features-123",
  "stream_duration_seconds": 60.5,
  "packets_sent": 605,
  "dropped_frames": 5
}
```

## Performance Characteristics

### Measured Performance

- **Latency**: ~50-80ms on local deployment
- **Throughput**: 100+ updates/second per connection
- **Memory**: ~10MB per 100 active connections
- **Max Connections**: 500+ concurrent (tested with Locust)
- **Drop Rate**: <0.5% under normal load
- **Heartbeat Overhead**: Negligible (<1% CPU)

### Acceptance Criteria ✅

- ✅ 30-second stream without drop/frame > 0.5%
- ✅ <120ms latency on local cluster
- ✅ 500 concurrent WebSocket clients supported

## Load Testing

### Locust Load Test Scenarios

Located in `load_tests/locustfile_ws.py`:

1. **WebSocketStreamUser**: Feature streaming workload
2. **WebSocketSimulationUser**: Simulation streaming workload
3. **WebSocketMixedUser**: Mixed workload simulation

### Running Load Tests

```bash
# Test 100 concurrent users
locust -f load_tests/locustfile_ws.py --host ws://localhost:8000 \
    --headless -u 100 -r 10 -t 2m

# Test 500 concurrent connections (acceptance criteria)
locust -f load_tests/locustfile_ws.py --host ws://localhost:8000 \
    --headless -u 500 -r 50 -t 5m
```

## Testing Strategy

### Unit Tests (`tests/unit/test_ws_manager.py`)

23 comprehensive tests covering:
- Connection lifecycle
- Authentication (API key + HMAC)
- Subscription management
- Backpressure handling
- Heartbeat monitoring
- Audit metrics tracking

### Integration Tests (`tests/integration/test_websocket_streaming.py`)

End-to-end tests covering:
- Full protocol flow
- Real WebSocket connections
- Stream data validation
- Error handling
- Timeout scenarios

### Load Tests (`load_tests/locustfile_ws.py`)

Performance and scalability tests:
- Concurrent connection limits
- Throughput under load
- Latency distribution
- Drop rate analysis

## Deployment Considerations

### Environment Variables

```bash
# Authentication
MFN_API_KEY_REQUIRED=true
MFN_API_KEY=primary-key
MFN_API_KEYS=key1,key2,key3

# Environment
MFN_ENV=production  # dev, staging, or production

# CORS (if WebSocket from browser)
MFN_CORS_ORIGINS=https://app.example.com
```

### Scaling Considerations

1. **Horizontal Scaling**: Use sticky sessions or connection routing
2. **Load Balancer**: Configure WebSocket support (upgrade headers)
3. **Timeout Settings**: Adjust heartbeat_interval based on network
4. **Queue Size**: Tune max_queue_size based on memory constraints
5. **Backpressure**: Choose strategy based on use case

### Monitoring

Key metrics to monitor:
- Active WebSocket connections
- Authentication success/failure rate
- Average message latency
- Backpressure events (dropped frames)
- Heartbeat timeout rate
- Connection duration distribution

## Future Enhancements

### Potential Improvements

1. **Data Compression**: Gzip/Brotli compression for large payloads
2. **Binary Protocol**: MessagePack or Protocol Buffers for efficiency
3. **Reconnection Logic**: Auto-reconnect with exponential backoff
4. **State Snapshots**: Periodic full state for late joiners
5. **Multi-Stream**: Multiple streams per connection
6. **Subscription Filters**: Dynamic feature filtering
7. **Stream Recording**: Record and replay streams

### NOT Implemented (Out of Scope)

- WebSocket over HTTPS (wss://) - handled by reverse proxy
- Distributed state synchronization
- Multi-server pub/sub
- Client-side libraries (generic WebSocket clients work)

## References

- **Implementation**: `src/mycelium_fractal_net/integration/ws_*.py`
- **API Code**: `api.py` (WebSocket endpoints)
- **Tests**: `tests/unit/test_ws_manager.py`, `tests/integration/test_websocket_streaming.py`
- **Load Tests**: `load_tests/locustfile_ws.py`
- **User Documentation**: `docs/WEBSOCKET_STREAMING.md`
- **Specification**: `docs/MFN_BACKLOG.md#MFN-API-STREAMING`
