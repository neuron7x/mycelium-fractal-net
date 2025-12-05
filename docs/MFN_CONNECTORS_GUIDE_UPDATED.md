# MyceliumFractalNet Connectors & Publishers Guide

**Version**: 4.1.0  
**Updated**: 2025-12-05

---

## Overview

MyceliumFractalNet provides a comprehensive integration layer for connecting to external systems:

- **Connectors** (Upstream): Pull data from external sources
- **Publishers** (Downstream): Push results to external systems

All connectors and publishers implement:
- ✅ Retry logic with exponential backoff
- ✅ Structured logging
- ✅ Metrics collection
- ✅ Graceful error handling
- ✅ Connection pooling and reuse

---

## Connectors (Upstream Data Sources)

### REST API Connector

Pull data from external HTTP APIs.

**Features**:
- Async HTTP client (aiohttp)
- Request retries with backoff
- Authentication support
- Rate limiting awareness

**Usage**:

```python
from mycelium_fractal_net.integration.connectors import RESTConnector, ConnectorConfig

# Create connector
config = ConnectorConfig(
    max_retries=3,
    timeout=30.0,
    enabled=True
)

connector = RESTConnector(
    base_url="https://api.example.com",
    config=config,
    headers={"Authorization": "Bearer token"}
)

# Connect
await connector.connect()

# Fetch data
data = await connector.fetch("/data/latest")
print(f"Received: {data}")

# Disconnect
await connector.disconnect()
```

**Advanced Example** (with retry):

```python
import asyncio

async def fetch_with_retry():
    connector = RESTConnector("https://api.example.com")
    await connector.connect()
    
    try:
        # Fetch with automatic retry
        data = await connector.fetch(
            "/simulation/params",
            params={"seed": 42, "grid_size": 64}
        )
        return data
    except Exception as e:
        print(f"Failed after retries: {e}")
    finally:
        await connector.disconnect()

result = asyncio.run(fetch_with_retry())
```

---

### Kafka Consumer Connector

Consume messages from Kafka topics.

**Features**:
- Automatic offset management
- Consumer group support
- Batch processing
- Error handling and DLQ

**Usage**:

```python
from mycelium_fractal_net.integration.connectors import KafkaConnector, ConnectorConfig

# Create connector
connector = KafkaConnector(
    bootstrap_servers=["localhost:9092"],
    topic="mfn-input",
    group_id="mfn-consumer-group",
    config=ConnectorConfig(enabled=True)
)

# Connect
await connector.connect()

# Consume messages
async for message in connector.consume():
    print(f"Received: {message}")
    
    # Process message
    params = message["params"]
    result = run_simulation(params)
    
    # Commit offset
    await connector.commit()

# Disconnect
await connector.disconnect()
```

**Batch Processing**:

```python
async def batch_consume():
    connector = KafkaConnector(
        bootstrap_servers=["localhost:9092"],
        topic="mfn-batch-input",
        group_id="mfn-batch-processor"
    )
    await connector.connect()
    
    batch = []
    async for message in connector.consume():
        batch.append(message)
        
        # Process in batches of 100
        if len(batch) >= 100:
            results = process_batch(batch)
            await publish_results(results)
            batch = []
            await connector.commit()
```

---

### File Feed Connector

Watch and process files from a directory.

**Features**:
- File system watching
- Pattern matching (glob)
- Automatic file archiving
- Concurrent processing

**Usage**:

```python
from mycelium_fractal_net.integration.connectors import FileFeedConnector

# Create connector
connector = FileFeedConnector(
    watch_dir="/data/input",
    pattern="*.json",
    archive_dir="/data/processed"
)

# Connect (starts watching)
await connector.connect()

# Process new files
async for file_path in connector.watch():
    print(f"New file: {file_path}")
    
    # Read and process
    with open(file_path) as f:
        data = json.load(f)
    
    result = process_data(data)
    
    # File is automatically archived after processing
```

---

## Publishers (Downstream Event Sinks)

### Webhook Publisher

Send results to HTTP endpoints.

**Features**:
- POST/PUT support
- Retry logic
- Batch publishing
- Request signing

**Usage**:

```python
from mycelium_fractal_net.integration.publishers import WebhookPublisher, PublisherConfig

# Create publisher
publisher = WebhookPublisher(
    webhook_url="https://api.example.com/webhook",
    config=PublisherConfig(
        max_retries=3,
        batch_size=10,
        enabled=True
    ),
    headers={"X-API-Key": "secret"}
)

# Connect
await publisher.connect()

# Publish single event
await publisher.publish({
    "event": "simulation_complete",
    "data": {
        "seed": 42,
        "fractal_dimension": 1.78,
        "growth_events": 23
    }
})

# Publish batch
events = [
    {"event": "validation", "loss": 0.1},
    {"event": "validation", "loss": 0.05},
]
await publisher.publish_batch(events)

# Disconnect
await publisher.disconnect()
```

**With Request Signing**:

```python
import hashlib
import hmac
import time

def sign_request(payload, secret):
    timestamp = str(int(time.time()))
    message = f"{timestamp}.{json.dumps(payload)}"
    signature = hmac.new(
        secret.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    return {"X-Timestamp": timestamp, "X-Signature": signature}

publisher = WebhookPublisher(
    webhook_url="https://api.example.com/webhook",
    headers_func=lambda payload: sign_request(payload, "secret")
)
```

---

### Kafka Producer Publisher

Publish events to Kafka topics.

**Features**:
- Async production
- Partitioning support
- Compression (gzip, snappy)
- Transaction support

**Usage**:

```python
from mycelium_fractal_net.integration.publishers import KafkaPublisher

# Create publisher
publisher = KafkaPublisher(
    bootstrap_servers=["localhost:9092"],
    topic="mfn-results",
    config=PublisherConfig(enabled=True)
)

# Connect
await publisher.connect()

# Publish event
await publisher.publish({
    "simulation_id": "sim-001",
    "result": {
        "fractal_dimension": 1.78,
        "field_stats": {"mean": -0.065, "std": 0.012}
    }
})

# Publish with custom key (for partitioning)
await publisher.publish(
    data={"result": "..."},
    key="simulation-group-A"
)

# Disconnect
await publisher.disconnect()
```

**High-Throughput Publishing**:

```python
async def publish_high_throughput():
    publisher = KafkaPublisher(
        bootstrap_servers=["kafka-1:9092", "kafka-2:9092"],
        topic="mfn-high-throughput",
        config=PublisherConfig(
            batch_size=1000,
            timeout=60.0
        )
    )
    await publisher.connect()
    
    # Publish 10K events
    events = [generate_event(i) for i in range(10000)]
    
    for event in events:
        await publisher.publish(event)
    
    # Flush remaining
    await publisher.flush()
    await publisher.disconnect()
```

---

### File Publisher

Write results to files.

**Features**:
- JSON/Parquet/CSV formats
- Atomic writes
- File rotation
- Compression

**Usage**:

```python
from mycelium_fractal_net.integration.publishers import FilePublisher

# Create publisher
publisher = FilePublisher(
    output_dir="/data/output",
    format="json",  # or "parquet", "csv"
    config=PublisherConfig(enabled=True)
)

# Connect
await publisher.connect()

# Publish data
await publisher.publish({
    "simulation_id": "sim-001",
    "results": [...]
})

# Files are automatically named with timestamps
# e.g., /data/output/results_20251205_123456.json

# Disconnect
await publisher.disconnect()
```

**Parquet with Schema**:

```python
import pandas as pd

publisher = FilePublisher(
    output_dir="/data/output",
    format="parquet"
)

await publisher.connect()

# Publish DataFrame
df = pd.DataFrame({
    "simulation_id": ["sim-001", "sim-002"],
    "fractal_dimension": [1.78, 1.82],
    "growth_events": [23, 19]
})

await publisher.publish(df)
```

---

## End-to-End Pipeline Example

Complete data pipeline with connector → processing → publisher:

```python
import asyncio
from mycelium_fractal_net.integration.connectors import KafkaConnector
from mycelium_fractal_net.integration.publishers import WebhookPublisher
from mycelium_fractal_net import run_mycelium_simulation_with_history
from mycelium_fractal_net.types import SimulationConfig

async def processing_pipeline():
    # Setup connector and publisher
    connector = KafkaConnector(
        bootstrap_servers=["localhost:9092"],
        topic="mfn-simulation-requests",
        group_id="mfn-processor"
    )
    
    publisher = WebhookPublisher(
        webhook_url="https://api.example.com/results",
        headers={"X-API-Key": "secret"}
    )
    
    # Connect
    await connector.connect()
    await publisher.connect()
    
    try:
        # Process messages
        async for message in connector.consume():
            print(f"Processing request: {message['request_id']}")
            
            # Create config from message
            config = SimulationConfig(
                seed=message["seed"],
                grid_size=message.get("grid_size", 64),
                steps=message.get("steps", 100)
            )
            
            # Run simulation
            result = run_mycelium_simulation_with_history(config)
            
            # Publish result
            await publisher.publish({
                "request_id": message["request_id"],
                "status": "complete",
                "results": {
                    "fractal_dimension": result.fractal_dimension,
                    "growth_events": result.growth_events,
                    "field_stats": result.field_stats
                }
            })
            
            # Commit offset
            await connector.commit()
            
    finally:
        await connector.disconnect()
        await publisher.disconnect()

# Run pipeline
asyncio.run(processing_pipeline())
```

---

## Configuration Best Practices

### 1. Environment-Based Configuration

```python
import os

# Load from environment
kafka_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(",")
webhook_url = os.getenv("WEBHOOK_URL", "https://default.example.com/webhook")
api_key = os.getenv("API_KEY", "")

connector = KafkaConnector(bootstrap_servers=kafka_servers)
publisher = WebhookPublisher(webhook_url=webhook_url, headers={"X-API-Key": api_key})
```

### 2. Retry Configuration

```python
config = ConnectorConfig(
    max_retries=5,
    retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    initial_retry_delay=1.0,
    max_retry_delay=60.0,
    timeout=30.0
)
```

### 3. Monitoring and Metrics

```python
# Get connector metrics
metrics = connector.get_metrics()
print(f"Messages consumed: {metrics.messages_received}")
print(f"Errors: {metrics.errors}")

# Get publisher metrics
metrics = publisher.get_metrics()
print(f"Messages published: {metrics.messages_sent}")
print(f"Success rate: {metrics.success_rate:.2%}")
```

### 4. Graceful Shutdown

```python
import signal

shutdown_event = asyncio.Event()

def signal_handler(sig, frame):
    shutdown_event.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

async def run_with_shutdown():
    connector = KafkaConnector(...)
    publisher = WebhookPublisher(...)
    
    await connector.connect()
    await publisher.connect()
    
    try:
        async for message in connector.consume():
            if shutdown_event.is_set():
                break
            
            # Process...
    finally:
        await connector.disconnect()
        await publisher.disconnect()
```

---

## Kafka Integration Setup

### Start Kafka (Docker Compose)

```yaml
version: '3'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
  
  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
```

```bash
docker-compose up -d
```

### Create Topics

```bash
# Create input topic
kafka-topics --create --topic mfn-simulation-requests \
  --bootstrap-server localhost:9092 \
  --partitions 10 \
  --replication-factor 1

# Create output topic
kafka-topics --create --topic mfn-results \
  --bootstrap-server localhost:9092 \
  --partitions 10 \
  --replication-factor 1
```

---

## Testing Connectors & Publishers

### Unit Tests

```python
import pytest
from mycelium_fractal_net.integration.connectors import RESTConnector

@pytest.mark.asyncio
async def test_rest_connector():
    connector = RESTConnector("https://httpbin.org")
    await connector.connect()
    
    data = await connector.fetch("/get")
    assert data is not None
    
    await connector.disconnect()
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_kafka_pipeline():
    # Setup
    producer = KafkaPublisher(
        bootstrap_servers=["localhost:9092"],
        topic="test-topic"
    )
    consumer = KafkaConnector(
        bootstrap_servers=["localhost:9092"],
        topic="test-topic",
        group_id="test-group"
    )
    
    await producer.connect()
    await consumer.connect()
    
    # Publish
    test_data = {"test": "message"}
    await producer.publish(test_data)
    
    # Consume
    received = None
    async for message in consumer.consume():
        received = message
        break
    
    assert received == test_data
    
    # Cleanup
    await producer.disconnect()
    await consumer.disconnect()
```

---

## Troubleshooting

### Common Issues

**1. Kafka Connection Refused**

```
Error: Connection refused (localhost:9092)
```

Solution: Check Kafka is running and accessible:

```bash
docker ps | grep kafka
kafka-topics --list --bootstrap-server localhost:9092
```

**2. HTTP Timeouts**

```
Error: Request timeout after 30.0 seconds
```

Solution: Increase timeout or check network connectivity:

```python
config = ConnectorConfig(timeout=60.0)
```

**3. Authentication Errors**

```
Error: 401 Unauthorized
```

Solution: Verify API keys/credentials:

```python
headers = {"X-API-Key": os.getenv("API_KEY")}
connector = RESTConnector(base_url="...", headers=headers)
```

---

## References

- [Kafka Python Documentation](https://kafka-python.readthedocs.io/)
- [aiohttp Documentation](https://docs.aiohttp.org/)
- [MFN Integration Spec](MFN_INTEGRATION_SPEC.md)
- [MFN System Role](MFN_SYSTEM_ROLE.md)
