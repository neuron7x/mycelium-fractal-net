"""
Downstream event publishers for MyceliumFractalNet.

Provides production-ready publishers for pushing simulation results and
feature vectors to external systems:
- Kafka producer for event-driven architectures
- Webhook publisher for HTTP callbacks
- Message queue integration (RabbitMQ/Redis)

All publishers support:
- Batch publishing for efficiency
- Delivery guarantees (at-least-once)
- Retry logic with exponential backoff
- Circuit breaker pattern
- Structured logging and metrics

Usage:
    >>> from mycelium_fractal_net.integration.publishers import KafkaPublisher
    >>> publisher = KafkaPublisher(
    ...     bootstrap_servers="localhost:9092",
    ...     topic="simulation-results"
    ... )
    >>> await publisher.publish({
    ...     "D_box": 1.584,
    ...     "V_mean": -67.5,
    ...     "f_active": 0.42
    ... })

Reference: docs/known_issues.md#ISSUE-002
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class PublisherStatus(str, Enum):
    """Status of publisher."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    CIRCUIT_OPEN = "circuit_open"


class DeliveryGuarantee(str, Enum):
    """Delivery guarantee level."""

    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


@dataclass
class PublisherConfig:
    """Configuration for event publishers."""

    delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE
    batch_size: int = 100
    batch_timeout: float = 5.0
    max_retries: int = 3
    initial_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    backoff_multiplier: float = 2.0
    timeout: float = 30.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    enable_metrics: bool = True
    enable_compression: bool = False


@dataclass
class PublisherMetrics:
    """Metrics for publisher operations."""

    messages_sent: int = 0
    messages_failed: int = 0
    bytes_sent: int = 0
    batches_sent: int = 0
    publish_errors: int = 0
    retry_count: int = 0
    last_success_time: Optional[datetime] = None
    last_error_time: Optional[datetime] = None
    last_error_message: Optional[str] = None


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(self, threshold: int = 5, timeout: float = 60.0):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.is_open = False

    def record_success(self) -> None:
        """Record successful operation."""
        self.failure_count = 0
        self.is_open = False

    def record_failure(self) -> None:
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.threshold:
            self.is_open = True
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )

    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        if not self.is_open:
            return True

        if self.last_failure_time is not None:
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.timeout:
                logger.info("Circuit breaker timeout expired, attempting to close")
                self.is_open = False
                self.failure_count = 0
                return True

        return False


class BasePublisher(ABC):
    """Base class for all event publishers."""

    def __init__(
        self,
        publisher_id: str,
        config: Optional[PublisherConfig] = None,
    ):
        self.publisher_id = publisher_id
        self.config = config or PublisherConfig()
        self.status = PublisherStatus.DISCONNECTED
        self.metrics = PublisherMetrics()
        self.circuit_breaker = CircuitBreaker(
            threshold=self.config.circuit_breaker_threshold,
            timeout=self.config.circuit_breaker_timeout,
        )
        self._batch: List[Dict[str, Any]] = []
        self._batch_lock = asyncio.Lock()
        self._batch_task: Optional[asyncio.Task] = None

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to destination."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to destination."""
        pass

    @abstractmethod
    async def _publish_single(self, message: Dict[str, Any]) -> None:
        """Publish single message (implemented by subclasses)."""
        pass

    async def publish(self, message: Dict[str, Any]) -> None:
        """Publish message with batching support."""
        async with self._batch_lock:
            self._batch.append(message)

            # Start batch timeout task if not already running
            if self._batch_task is None or self._batch_task.done():
                self._batch_task = asyncio.create_task(
                    self._flush_after_timeout()
                )

            # Flush if batch is full
            if len(self._batch) >= self.config.batch_size:
                await self._flush_batch()

    async def _flush_after_timeout(self) -> None:
        """Flush batch after timeout."""
        await asyncio.sleep(self.config.batch_timeout)
        async with self._batch_lock:
            if self._batch:
                await self._flush_batch()

    async def _flush_batch(self) -> None:
        """Flush pending batch."""
        if not self._batch:
            return

        batch = self._batch.copy()
        self._batch.clear()

        try:
            await self._publish_batch(batch)
            self.metrics.batches_sent += 1
        except Exception as e:
            logger.error(f"Failed to flush batch: {e}")
            # Re-add messages to batch for retry if configured
            if self.config.delivery_guarantee == DeliveryGuarantee.AT_LEAST_ONCE:
                self._batch.extend(batch)

    async def _publish_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Publish batch of messages."""
        for message in batch:
            await self._publish_single(message)

    async def retry_operation(
        self,
        operation: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute operation with retry logic."""
        if not self.circuit_breaker.can_execute():
            raise RuntimeError("Circuit breaker is open")

        retry_count = 0
        last_exception = None

        while retry_count <= self.config.max_retries:
            try:
                result = await operation(*args, **kwargs)
                self.circuit_breaker.record_success()
                return result

            except Exception as e:
                last_exception = e
                retry_count += 1
                self.metrics.retry_count += 1
                self.circuit_breaker.record_failure()

                if retry_count > self.config.max_retries:
                    self.metrics.publish_errors += 1
                    self.metrics.last_error_time = datetime.now()
                    self.metrics.last_error_message = str(e)
                    logger.error(
                        f"Publish failed after {self.config.max_retries} retries: {e}"
                    )
                    raise

                delay = self._calculate_retry_delay(retry_count)
                logger.warning(
                    f"Publish failed (attempt {retry_count}/{self.config.max_retries}), "
                    f"retrying in {delay:.2f}s: {e}"
                )
                await asyncio.sleep(delay)

        if last_exception:
            raise last_exception

    def _calculate_retry_delay(self, retry_count: int) -> float:
        """Calculate delay before next retry."""
        return min(
            self.config.initial_retry_delay * (self.config.backoff_multiplier ** (retry_count - 1)),
            self.config.max_retry_delay,
        )


class KafkaPublisher(BasePublisher):
    """
    Kafka producer publisher for event-driven architectures.
    
    Note: Requires aiokafka package (optional dependency).
    Install with: pip install aiokafka
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        config: Optional[PublisherConfig] = None,
        partitioner: Optional[str] = None,
    ):
        super().__init__(publisher_id=f"kafka-{topic}", config=config)
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.partitioner = partitioner
        self.producer: Optional[Any] = None

    async def connect(self) -> None:
        """Establish connection to Kafka cluster."""
        try:
            from aiokafka import AIOKafkaProducer
        except ImportError:
            raise ImportError(
                "aiokafka is required for Kafka publisher. "
                "Install with: pip install aiokafka"
            )

        self.status = PublisherStatus.CONNECTING
        logger.info(f"Connecting to Kafka: {self.bootstrap_servers}, topic: {self.topic}")

        async def _connect():
            acks_config = (
                "all"
                if self.config.delivery_guarantee == DeliveryGuarantee.AT_LEAST_ONCE
                else 1
            )
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks=acks_config,
            )
            await self.producer.start()

        await self.retry_operation(_connect)
        self.status = PublisherStatus.CONNECTED
        logger.info(f"Connected to Kafka topic: {self.topic}")

    async def disconnect(self) -> None:
        """Close connection to Kafka cluster."""
        # Flush remaining messages
        async with self._batch_lock:
            if self._batch:
                await self._flush_batch()

        if self.producer:
            await self.producer.stop()
            self.producer = None
        self.status = PublisherStatus.DISCONNECTED
        logger.info(f"Disconnected from Kafka topic: {self.topic}")

    async def _publish_single(self, message: Dict[str, Any]) -> None:
        """Publish single message to Kafka."""
        if not self.producer:
            await self.connect()

        async def _send():
            await self.producer.send_and_wait(self.topic, message)

        await self.retry_operation(_send)
        self.metrics.messages_sent += 1
        self.metrics.bytes_sent += len(json.dumps(message))
        self.metrics.last_success_time = datetime.now()
        
        logger.debug(f"Published message to Kafka topic: {self.topic}")


class WebhookPublisher(BasePublisher):
    """
    Webhook publisher for HTTP callbacks.
    
    Supports HMAC signature verification for security.
    """

    def __init__(
        self,
        url: str,
        secret_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        config: Optional[PublisherConfig] = None,
    ):
        super().__init__(publisher_id=f"webhook-{url}", config=config)
        self.url = url
        self.secret_key = secret_key
        self.headers = headers or {"Content-Type": "application/json"}
        self.session: Optional[Any] = None

    async def connect(self) -> None:
        """Create HTTP session."""
        try:
            import aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp is required for Webhook publisher. "
                "Install with: pip install aiohttp"
            )
        
        self.status = PublisherStatus.CONNECTING
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        self.status = PublisherStatus.CONNECTED
        logger.info(f"Connected to webhook: {self.url}")

    async def disconnect(self) -> None:
        """Close HTTP session."""
        # Flush remaining messages
        async with self._batch_lock:
            if self._batch:
                await self._flush_batch()

        if self.session:
            await self.session.close()
            self.session = None
        self.status = PublisherStatus.DISCONNECTED
        logger.info(f"Disconnected from webhook: {self.url}")

    def _sign_payload(self, payload: str) -> str:
        """Generate HMAC signature for payload."""
        if not self.secret_key:
            return ""
        
        signature = hmac.new(
            self.secret_key.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"sha256={signature}"

    async def _publish_single(self, message: Dict[str, Any]) -> None:
        """Publish single message via webhook."""
        if not self.session:
            await self.connect()

        async def _post():
            payload = json.dumps(message)
            headers = self.headers.copy()
            
            # Add HMAC signature if secret key is configured
            if self.secret_key:
                headers["X-Webhook-Signature"] = self._sign_payload(payload)
            
            async with self.session.post(
                self.url,
                data=payload,
                headers=headers,
            ) as response:
                response.raise_for_status()
                return await response.text()

        await self.retry_operation(_post)
        self.metrics.messages_sent += 1
        self.metrics.bytes_sent += len(json.dumps(message))
        self.metrics.last_success_time = datetime.now()
        
        logger.debug(f"Published message to webhook: {self.url}")


class RedisPublisher(BasePublisher):
    """
    Redis publisher for message queue integration.
    
    Note: Requires redis package (optional dependency).
    Install with: pip install redis aioredis
    """

    def __init__(
        self,
        redis_url: str,
        channel: str,
        config: Optional[PublisherConfig] = None,
    ):
        super().__init__(publisher_id=f"redis-{channel}", config=config)
        self.redis_url = redis_url
        self.channel = channel
        self.redis: Optional[Any] = None

    async def connect(self) -> None:
        """Establish connection to Redis."""
        try:
            import redis.asyncio as aioredis
        except ImportError:
            raise ImportError(
                "redis is required for Redis publisher. "
                "Install with: pip install redis"
            )

        self.status = PublisherStatus.CONNECTING
        logger.info(f"Connecting to Redis: {self.redis_url}, channel: {self.channel}")

        async def _connect():
            self.redis = await aioredis.from_url(self.redis_url)

        await self.retry_operation(_connect)
        self.status = PublisherStatus.CONNECTED
        logger.info(f"Connected to Redis channel: {self.channel}")

    async def disconnect(self) -> None:
        """Close connection to Redis."""
        # Flush remaining messages
        async with self._batch_lock:
            if self._batch:
                await self._flush_batch()

        if self.redis:
            await self.redis.close()
            self.redis = None
        self.status = PublisherStatus.DISCONNECTED
        logger.info(f"Disconnected from Redis channel: {self.channel}")

    async def _publish_single(self, message: Dict[str, Any]) -> None:
        """Publish single message to Redis."""
        if not self.redis:
            await self.connect()

        async def _publish():
            payload = json.dumps(message)
            await self.redis.publish(self.channel, payload)

        await self.retry_operation(_publish)
        self.metrics.messages_sent += 1
        self.metrics.bytes_sent += len(json.dumps(message))
        self.metrics.last_success_time = datetime.now()
        
        logger.debug(f"Published message to Redis channel: {self.channel}")


class RabbitMQPublisher(BasePublisher):
    """
    RabbitMQ publisher for message queue integration.
    
    Note: Requires aio-pika package (optional dependency).
    Install with: pip install aio-pika
    """

    def __init__(
        self,
        amqp_url: str,
        exchange: str,
        routing_key: str,
        config: Optional[PublisherConfig] = None,
    ):
        super().__init__(publisher_id=f"rabbitmq-{exchange}", config=config)
        self.amqp_url = amqp_url
        self.exchange = exchange
        self.routing_key = routing_key
        self.connection: Optional[Any] = None
        self.channel: Optional[Any] = None
        self._exchange: Optional[Any] = None
        self._aio_pika: Optional[Any] = None

    async def connect(self) -> None:
        """Establish connection to RabbitMQ."""
        try:
            import aio_pika
            
            # Store for later use
            self._aio_pika = aio_pika
        except ImportError:
            raise ImportError(
                "aio-pika is required for RabbitMQ publisher. "
                "Install with: pip install aio-pika"
            )

        self.status = PublisherStatus.CONNECTING
        logger.info(f"Connecting to RabbitMQ: {self.amqp_url}, exchange: {self.exchange}")

        async def _connect():
            self.connection = await aio_pika.connect_robust(self.amqp_url)
            self.channel = await self.connection.channel()
            # Declare and store exchange for later use
            self._exchange = await self.channel.declare_exchange(
                self.exchange,
                aio_pika.ExchangeType.TOPIC,
                durable=True,
            )

        await self.retry_operation(_connect)
        self.status = PublisherStatus.CONNECTED
        logger.info(f"Connected to RabbitMQ exchange: {self.exchange}")

    async def disconnect(self) -> None:
        """Close connection to RabbitMQ."""
        # Flush remaining messages
        async with self._batch_lock:
            if self._batch:
                await self._flush_batch()

        if self.connection:
            await self.connection.close()
            self.connection = None
            self.channel = None
        self.status = PublisherStatus.DISCONNECTED
        logger.info(f"Disconnected from RabbitMQ exchange: {self.exchange}")

    async def _publish_single(self, message: Dict[str, Any]) -> None:
        """Publish single message to RabbitMQ."""
        if not self.channel:
            await self.connect()

        async def _publish():
            payload = json.dumps(message).encode("utf-8")
            # Use the declared exchange, not default exchange
            await self._exchange.publish(
                self._aio_pika.Message(
                    body=payload,
                    delivery_mode=self._aio_pika.DeliveryMode.PERSISTENT,
                ),
                routing_key=self.routing_key,
            )

        await self.retry_operation(_publish)
        self.metrics.messages_sent += 1
        self.metrics.bytes_sent += len(json.dumps(message))
        self.metrics.last_success_time = datetime.now()
        
        logger.debug(f"Published message to RabbitMQ: {self.routing_key}")


# Example usage and testing functions
async def test_webhook_publisher():
    """Test webhook publisher."""
    publisher = WebhookPublisher(
        url="https://api.example.com/webhooks/mfn",
        secret_key="test-secret",
    )
    
    try:
        await publisher.connect()
        
        # Publish test message
        await publisher.publish({
            "D_box": 1.584,
            "V_mean": -67.5,
            "f_active": 0.42,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Allow batch to flush
        await asyncio.sleep(6.0)
        
    finally:
        await publisher.disconnect()


if __name__ == "__main__":
    # Example usage
    print("MFN Event Publishers - Production Ready")
    print("Available publishers:")
    print("  - KafkaPublisher (requires aiokafka)")
    print("  - WebhookPublisher")
    print("  - RedisPublisher (requires redis)")
    print("  - RabbitMQPublisher (requires aio-pika)")
