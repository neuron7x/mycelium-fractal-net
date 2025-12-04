# Integration Guide - MyceliumFractalNet

**Version**: 1.0  
**Last Updated**: 2025-12-04  
**Target**: Production deployments and external system integration

---

## Overview

This guide covers integrating MyceliumFractalNet (MFN) with external systems through data connectors (upstream) and event publishers (downstream). MFN provides production-ready integration components with retry logic, circuit breakers, and comprehensive observability.

**Integration Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    External Systems                          │
├──────────────┬──────────────┬──────────────┬────────────────┤
│   Kafka      │  REST APIs   │   Files      │   Databases    │
└──────┬───────┴──────┬───────┴──────┬───────┴──────┬─────────┘
       │              │              │              │
       │         Data Connectors (Upstream)         │
       │              │              │              │
       └──────────────┴──────────────┴──────────────┘
                      │
         ┌────────────▼────────────┐
         │  MyceliumFractalNet     │
         │  - Simulation           │
         │  - Feature Extraction   │
         │  - Analysis             │
         └────────────┬────────────┘
                      │
       ┌──────────────┴──────────────┬──────────────┐
       │                             │              │
  Event Publishers (Downstream)     │              │
       │                             │              │
┌──────▼───────┬──────▼───────┬─────▼──────┬───────▼─────┐
│   Kafka      │  Webhooks    │   Redis    │  RabbitMQ   │
└──────────────┴──────────────┴────────────┴─────────────┘
       │              │              │              │
┌──────▼──────────────▼──────────────▼──────────────▼─────┐
│            ML Models / Analytics / Dashboards            │
└──────────────────────────────────────────────────────────┘
```

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Connectors (Upstream)](#data-connectors-upstream)
3. [Event Publishers (Downstream)](#event-publishers-downstream)
4. [Configuration](#configuration)
5. [Error Handling & Resilience](#error-handling--resilience)
6. [Observability](#observability)
7. [Production Deployment](#production-deployment)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

Install MFN with optional dependencies for your integration:

```bash
# Core installation
pip install mycelium-fractal-net

# With Kafka support
pip install mycelium-fractal-net aiokafka

# With Redis support
pip install mycelium-fractal-net redis

# With RabbitMQ support
pip install mycelium-fractal-net aio-pika

# All integrations
pip install mycelium-fractal-net aiokafka redis aio-pika aiohttp
```

### Basic Example

```python
import asyncio
from mycelium_fractal_net.integration import (
    FileFeedConnector,
    WebhookPublisher,
)
from mycelium_fractal_net import (
    run_mycelium_simulation_with_history,
    compute_fractal_features,
)

async def process_pipeline():
    # Setup connector and publisher
    connector = FileFeedConnector(
        directory="/data/simulation-params",
        file_pattern="*.json",
    )
    
    publisher = WebhookPublisher(
        url="https://api.example.com/results",
        secret_key="your-secret-key",
    )
    
    await connector.connect()
    await publisher.connect()
    
    try:
        # Consume simulation parameters
        async for params in connector.consume():
            # Run simulation
            result = run_mycelium_simulation_with_history(params)
            
            # Extract features
            features = compute_fractal_features(result)
            
            # Publish results
            await publisher.publish(features.to_dict())
            
    finally:
        await connector.disconnect()
        await publisher.disconnect()

# Run pipeline
asyncio.run(process_pipeline())
```

---

## Data Connectors (Upstream)

Data connectors ingest simulation parameters and configuration from external systems.

### Kafka Connector

**Use Case**: Real-time event streaming, microservices architecture

**Installation**: `pip install aiokafka`

```python
from mycelium_fractal_net.integration import (
    KafkaConnector,
    ConnectorConfig,
    RetryStrategy,
)

# Configure connector
config = ConnectorConfig(
    retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    max_retries=3,
    initial_retry_delay=1.0,
    circuit_breaker_threshold=5,
)

# Create connector
connector = KafkaConnector(
    bootstrap_servers="kafka1:9092,kafka2:9092",
    topic="mfn-simulation-params",
    group_id="mfn-consumer-group",
    auto_offset_reset="earliest",
    config=config,
)

await connector.connect()

# Consume messages
async for message in connector.consume():
    simulation_params = message  # Already deserialized as JSON
    print(f"Received params: {simulation_params}")
    
    # Check metrics
    print(f"Messages received: {connector.metrics.messages_received}")
    print(f"Retry count: {connector.metrics.retry_count}")

await connector.disconnect()
```

**Configuration Options:**
- `bootstrap_servers`: Kafka broker addresses
- `topic`: Topic to consume from
- `group_id`: Consumer group ID
- `auto_offset_reset`: "earliest" or "latest"
- `enable_auto_commit`: Automatic offset commit

### File Feed Connector

**Use Case**: Batch processing, file-based data ingestion

**No additional dependencies required**

```python
from mycelium_fractal_net.integration import FileFeedConnector
from pathlib import Path

# One-time batch processing
connector = FileFeedConnector(
    directory="/data/batch-input",
    file_pattern="*.json",
    watch_mode=False,  # Process existing files once
)

await connector.connect()

async for data in connector.consume():
    print(f"Processing file: {data}")

await connector.disconnect()

# Continuous monitoring
connector_watch = FileFeedConnector(
    directory="/data/hot-folder",
    file_pattern="params_*.json",
    watch_mode=True,  # Watch for new files
)

await connector_watch.connect()

async for data in connector_watch.consume():
    print(f"New file detected: {data}")
    # Process data
```

**Supported Formats:**
- JSON files (automatic parsing)
- Other formats (returns raw content)

### REST API Connector

**Use Case**: Polling external APIs, scheduled data retrieval

**Installation**: `pip install aiohttp`

```python
from mycelium_fractal_net.integration import RestApiConnector

connector = RestApiConnector(
    base_url="https://api.example.com",
    endpoint="/v1/simulation-configs",
    poll_interval=30.0,  # Poll every 30 seconds
    headers={
        "Authorization": "Bearer your-token",
        "Content-Type": "application/json",
    },
)

await connector.connect()

async for data in connector.consume():
    config = data
    print(f"Fetched config: {config}")
    
    # Process immediately or break after first fetch
    break

await connector.disconnect()
```

**Configuration Options:**
- `base_url`: Base URL of API
- `endpoint`: API endpoint path
- `poll_interval`: Seconds between polls
- `headers`: HTTP headers for authentication

### Database Connector

**Use Case**: Extract data from databases

**Base class - extend for specific databases**

```python
from mycelium_fractal_net.integration import DatabaseConnector
import asyncpg

class PostgresConnector(DatabaseConnector):
    async def connect(self):
        self.connection = await asyncpg.connect(self.connection_string)
        self.status = ConnectorStatus.CONNECTED
    
    async def consume(self):
        while True:
            rows = await self.connection.fetch(self.query)
            yield [dict(row) for row in rows]
            await asyncio.sleep(self.poll_interval)

# Usage
connector = PostgresConnector(
    connection_string="postgresql://user:pass@localhost/mfn",
    query="SELECT * FROM simulation_params WHERE status = 'pending'",
    poll_interval=60.0,
)
```

---

## Event Publishers (Downstream)

Event publishers send simulation results and feature vectors to external systems.

### Kafka Publisher

**Use Case**: Event-driven architectures, microservices

**Installation**: `pip install aiokafka`

```python
from mycelium_fractal_net.integration import (
    KafkaPublisher,
    PublisherConfig,
    DeliveryGuarantee,
)

# Configure publisher
config = PublisherConfig(
    delivery_guarantee=DeliveryGuarantee.AT_LEAST_ONCE,
    batch_size=100,
    batch_timeout=5.0,
    max_retries=3,
)

# Create publisher
publisher = KafkaPublisher(
    bootstrap_servers="kafka1:9092,kafka2:9092",
    topic="mfn-results",
    config=config,
)

await publisher.connect()

# Publish messages
for result in simulation_results:
    await publisher.publish({
        "simulation_id": result.id,
        "D_box": result.D_box,
        "V_mean": result.V_mean,
        "timestamp": datetime.now().isoformat(),
    })

# Flush pending messages
await publisher.disconnect()  # Automatically flushes

print(f"Messages sent: {publisher.metrics.messages_sent}")
print(f"Batches sent: {publisher.metrics.batches_sent}")
```

### Webhook Publisher

**Use Case**: HTTP callbacks, event notifications

**Installation**: `pip install aiohttp`

```python
from mycelium_fractal_net.integration import WebhookPublisher

# With HMAC signature verification
publisher = WebhookPublisher(
    url="https://api.example.com/webhooks/mfn",
    secret_key="your-webhook-secret",
    headers={"X-Custom-Header": "value"},
)

await publisher.connect()

# Publish results
await publisher.publish({
    "event": "simulation_complete",
    "data": {
        "D_box": 1.584,
        "V_mean": -67.5,
        "f_active": 0.42,
    },
})

await publisher.disconnect()
```

**HMAC Signature Verification** (Receiver side):

```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    expected = hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    received = signature.replace('sha256=', '')
    return hmac.compare_digest(expected, received)

# In your webhook handler
@app.post("/webhooks/mfn")
async def webhook_handler(request):
    payload = await request.body()
    signature = request.headers.get("X-Webhook-Signature")
    
    if not verify_webhook(payload, signature, SECRET_KEY):
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    # Process webhook
```

### Redis Publisher

**Use Case**: Pub/sub messaging, real-time updates

**Installation**: `pip install redis`

```python
from mycelium_fractal_net.integration import RedisPublisher

publisher = RedisPublisher(
    redis_url="redis://localhost:6379",
    channel="mfn:results",
)

await publisher.connect()

# Publish to channel
await publisher.publish({
    "result_id": "sim-123",
    "features": {...},
})

await publisher.disconnect()
```

### RabbitMQ Publisher

**Use Case**: Message queuing, work distribution

**Installation**: `pip install aio-pika`

```python
from mycelium_fractal_net.integration import RabbitMQPublisher

publisher = RabbitMQPublisher(
    amqp_url="amqp://user:pass@localhost:5672/",
    exchange="mfn-results",
    routing_key="simulation.complete",
)

await publisher.connect()

# Publish message
await publisher.publish({
    "simulation_id": "sim-456",
    "status": "complete",
    "features": {...},
})

await publisher.disconnect()
```

---

## Configuration

### Connector Configuration

```python
from mycelium_fractal_net.integration import ConnectorConfig, RetryStrategy

config = ConnectorConfig(
    # Retry strategy
    retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    max_retries=3,
    initial_retry_delay=1.0,
    max_retry_delay=60.0,
    backoff_multiplier=2.0,
    
    # Timeouts
    timeout=30.0,
    
    # Circuit breaker
    circuit_breaker_threshold=5,
    circuit_breaker_timeout=60.0,
    
    # Metrics
    enable_metrics=True,
)
```

### Publisher Configuration

```python
from mycelium_fractal_net.integration import PublisherConfig, DeliveryGuarantee

config = PublisherConfig(
    # Delivery
    delivery_guarantee=DeliveryGuarantee.AT_LEAST_ONCE,
    
    # Batching
    batch_size=100,
    batch_timeout=5.0,
    
    # Retry
    max_retries=3,
    initial_retry_delay=1.0,
    max_retry_delay=60.0,
    backoff_multiplier=2.0,
    
    # Timeouts
    timeout=30.0,
    
    # Circuit breaker
    circuit_breaker_threshold=5,
    circuit_breaker_timeout=60.0,
    
    # Features
    enable_metrics=True,
    enable_compression=False,
)
```

---

## Error Handling & Resilience

### Retry Logic

All connectors and publishers support configurable retry strategies:

```python
# Exponential backoff: 1s, 2s, 4s, 8s, ...
retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF

# Linear backoff: 1s, 2s, 3s, 4s, ...
retry_strategy=RetryStrategy.LINEAR_BACKOFF

# Immediate retry
retry_strategy=RetryStrategy.IMMEDIATE

# No retry
retry_strategy=RetryStrategy.NO_RETRY
```

### Circuit Breaker

Circuit breaker pattern prevents cascading failures:

```python
config = ConnectorConfig(
    circuit_breaker_threshold=5,  # Open after 5 failures
    circuit_breaker_timeout=60.0,  # Try again after 60s
)

# Circuit states:
# - CLOSED: Normal operation
# - OPEN: After threshold failures, blocks all requests
# - HALF-OPEN: After timeout, allows test request
```

### Error Handling Example

```python
try:
    async for message in connector.consume():
        await process_message(message)
except RuntimeError as e:
    if "Circuit breaker is open" in str(e):
        logger.warning("Circuit breaker open, backing off")
        await asyncio.sleep(60)
    else:
        raise
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Check metrics for diagnosis
    logger.info(f"Failed messages: {connector.metrics.messages_failed}")
    logger.info(f"Last error: {connector.metrics.last_error_message}")
```

---

## Observability

### Metrics

All connectors and publishers expose metrics:

```python
# Connector metrics
print(f"Messages received: {connector.metrics.messages_received}")
print(f"Messages failed: {connector.metrics.messages_failed}")
print(f"Bytes received: {connector.metrics.bytes_received}")
print(f"Connection errors: {connector.metrics.connection_errors}")
print(f"Retry count: {connector.metrics.retry_count}")
print(f"Last success: {connector.metrics.last_success_time}")
print(f"Last error: {connector.metrics.last_error_message}")

# Publisher metrics
print(f"Messages sent: {publisher.metrics.messages_sent}")
print(f"Messages failed: {publisher.metrics.messages_failed}")
print(f"Bytes sent: {publisher.metrics.bytes_sent}")
print(f"Batches sent: {publisher.metrics.batches_sent}")
print(f"Publish errors: {publisher.metrics.publish_errors}")
```

### Prometheus Integration

Export metrics to Prometheus:

```python
from prometheus_client import Counter, Gauge

# Define metrics
messages_received = Counter('mfn_connector_messages_received', 'Messages received')
messages_failed = Counter('mfn_connector_messages_failed', 'Messages failed')

# Update metrics
async for message in connector.consume():
    messages_received.inc()
    try:
        await process_message(message)
    except Exception:
        messages_failed.inc()
```

### Logging

All components use structured logging:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Connector/Publisher logs include:
# - Component ID
# - Operation type
# - Status changes
# - Error messages
# - Metrics updates
```

---

## Production Deployment

### Best Practices

1. **Use Environment Variables**:
   ```bash
   export KAFKA_BOOTSTRAP_SERVERS="kafka1:9092,kafka2:9092"
   export KAFKA_TOPIC="mfn-params"
   export WEBHOOK_URL="https://api.example.com/webhooks"
   export WEBHOOK_SECRET="your-secret"
   ```

2. **Configure Timeouts**:
   ```python
   config = ConnectorConfig(
       timeout=30.0,  # Connection timeout
       max_retry_delay=60.0,  # Max retry wait
   )
   ```

3. **Monitor Circuit Breakers**:
   ```python
   if connector.circuit_breaker.is_open:
       alert("Circuit breaker open for connector")
   ```

4. **Set Reasonable Batch Sizes**:
   ```python
   config = PublisherConfig(
       batch_size=100,  # Balance latency vs throughput
       batch_timeout=5.0,  # Max wait before flush
   )
   ```

5. **Use Health Checks**:
   ```python
   @app.get("/health")
   async def health():
       return {
           "connector": connector.status.value,
           "publisher": publisher.status.value,
           "circuit_breaker": not connector.circuit_breaker.is_open,
       }
   ```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mfn-pipeline
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: mfn
        image: mfn:latest
        env:
        - name: KAFKA_BOOTSTRAP_SERVERS
          value: "kafka-service:9092"
        - name: KAFKA_TOPIC
          valueFrom:
            configMapKeyRef:
              name: mfn-config
              key: kafka-topic
        - name: WEBHOOK_SECRET
          valueFrom:
            secretKeyRef:
              name: mfn-secrets
              key: webhook-secret
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

---

## Troubleshooting

### Common Issues

**1. Connection Timeouts**

```
Error: Connection timeout after 30.0s
```

**Solution**: Increase timeout or check network connectivity
```python
config = ConnectorConfig(timeout=60.0)
```

**2. Circuit Breaker Open**

```
Error: Circuit breaker is open
```

**Solution**: Wait for timeout or investigate root cause
```python
# Check error history
print(connector.metrics.last_error_message)
print(connector.metrics.connection_errors)

# Reset circuit breaker (if issue resolved)
connector.circuit_breaker.record_success()
```

**3. Message Deserialization Errors**

```
Error: JSON decode error
```

**Solution**: Validate message format
```python
try:
    async for message in connector.consume():
        # Validate before processing
        if not isinstance(message, dict):
            logger.warning(f"Invalid message format: {message}")
            continue
except json.JSONDecodeError as e:
    logger.error(f"JSON decode error: {e}")
```

**4. Memory Leaks with Batching**

```
Warning: Memory usage increasing
```

**Solution**: Reduce batch size or timeout
```python
config = PublisherConfig(
    batch_size=50,  # Reduce from 100
    batch_timeout=2.0,  # Flush more frequently
)
```

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Connector/Publisher will log:
# - Connection attempts
# - Retry operations
# - Circuit breaker state changes
# - Message processing
# - Error details
```

---

## Support & Resources

- **Documentation**: [docs/known_issues.md](known_issues.md)
- **API Reference**: [docs/ARCHITECTURE.md](ARCHITECTURE.md)
- **GitHub Issues**: https://github.com/neuron7x/mycelium-fractal-net/issues

---

**Last Updated**: 2025-12-04  
**Version**: 1.0  
**Maintainer**: MFN Integration Team
