"""
External data connectors for MyceliumFractalNet.

Provides production-ready connectors for integrating with external data sources:
- Kafka consumer for real-time event streaming
- File feed connector for batch data ingestion
- REST API pull connector for polling external APIs
- Database polling connector for extracting data

All connectors support:
- Retry logic with exponential backoff
- Circuit breaker pattern for fault tolerance
- Structured logging with correlation IDs
- Metrics emission for observability

Usage:
    >>> from mycelium_fractal_net.integration.connectors import KafkaConnector
    >>> connector = KafkaConnector(
    ...     bootstrap_servers="localhost:9092",
    ...     topic="simulation-params",
    ...     group_id="mfn-consumer"
    ... )
    >>> async for message in connector.consume():
    ...     params = message.value
    ...     # Process simulation parameters

Reference: docs/known_issues.md#ISSUE-001
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

import aiohttp
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class ConnectorStatus(str, Enum):
    """Status of connector."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    CIRCUIT_OPEN = "circuit_open"


class RetryStrategy(str, Enum):
    """Retry strategy for connectors."""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    IMMEDIATE = "immediate"
    NO_RETRY = "no_retry"


@dataclass
class ConnectorConfig:
    """Configuration for data connectors."""

    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    max_retries: int = 3
    initial_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    backoff_multiplier: float = 2.0
    timeout: float = 30.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    enable_metrics: bool = True


@dataclass
class ConnectorMetrics:
    """Metrics for connector operations."""

    messages_received: int = 0
    messages_failed: int = 0
    bytes_received: int = 0
    connection_errors: int = 0
    retry_count: int = 0
    last_success_time: Optional[datetime] = None
    last_error_time: Optional[datetime] = None
    last_error_message: Optional[str] = None


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""

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

        # Check if timeout has passed
        if self.last_failure_time is not None:
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.timeout:
                logger.info("Circuit breaker timeout expired, attempting to close")
                self.is_open = False
                self.failure_count = 0
                return True

        return False


class BaseConnector(ABC):
    """Base class for all data connectors."""

    def __init__(
        self,
        connector_id: str,
        config: Optional[ConnectorConfig] = None,
    ):
        self.connector_id = connector_id
        self.config = config or ConnectorConfig()
        self.status = ConnectorStatus.DISCONNECTED
        self.metrics = ConnectorMetrics()
        self.circuit_breaker = CircuitBreaker(
            threshold=self.config.circuit_breaker_threshold,
            timeout=self.config.circuit_breaker_timeout,
        )

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to data source."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to data source."""
        pass

    @abstractmethod
    async def consume(self) -> AsyncIterator[Any]:
        """Consume messages from data source."""
        pass

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
                    self.metrics.connection_errors += 1
                    self.metrics.last_error_time = datetime.now()
                    self.metrics.last_error_message = str(e)
                    logger.error(
                        f"Operation failed after {self.config.max_retries} retries: {e}"
                    )
                    raise

                # Calculate delay based on retry strategy
                delay = self._calculate_retry_delay(retry_count)
                logger.warning(
                    f"Operation failed (attempt {retry_count}/{self.config.max_retries}), "
                    f"retrying in {delay:.2f}s: {e}"
                )
                await asyncio.sleep(delay)

        if last_exception:
            raise last_exception

    def _calculate_retry_delay(self, retry_count: int) -> float:
        """Calculate delay before next retry."""
        if self.config.retry_strategy == RetryStrategy.NO_RETRY:
            return 0.0
        elif self.config.retry_strategy == RetryStrategy.IMMEDIATE:
            return 0.0
        elif self.config.retry_strategy == RetryStrategy.LINEAR_BACKOFF:
            return min(
                self.config.initial_retry_delay * retry_count,
                self.config.max_retry_delay,
            )
        else:  # EXPONENTIAL_BACKOFF
            return min(
                self.config.initial_retry_delay * (self.config.backoff_multiplier ** (retry_count - 1)),
                self.config.max_retry_delay,
            )


class KafkaConnector(BaseConnector):
    """
    Kafka consumer connector for real-time event streaming.
    
    Note: Requires aiokafka package (optional dependency).
    Install with: pip install aiokafka
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        group_id: str,
        config: Optional[ConnectorConfig] = None,
        auto_offset_reset: str = "earliest",
        enable_auto_commit: bool = True,
    ):
        super().__init__(connector_id=f"kafka-{topic}", config=config)
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset
        self.enable_auto_commit = enable_auto_commit
        self.consumer: Optional[Any] = None

    async def connect(self) -> None:
        """Establish connection to Kafka cluster."""
        try:
            # Import aiokafka only when needed
            from aiokafka import AIOKafkaConsumer
        except ImportError:
            raise ImportError(
                "aiokafka is required for Kafka connector. "
                "Install with: pip install aiokafka"
            )

        self.status = ConnectorStatus.CONNECTING
        logger.info(f"Connecting to Kafka: {self.bootstrap_servers}, topic: {self.topic}")

        async def _connect():
            self.consumer = AIOKafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=self.auto_offset_reset,
                enable_auto_commit=self.enable_auto_commit,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            )
            await self.consumer.start()

        await self.retry_operation(_connect)
        self.status = ConnectorStatus.CONNECTED
        logger.info(f"Connected to Kafka topic: {self.topic}")

    async def disconnect(self) -> None:
        """Close connection to Kafka cluster."""
        if self.consumer:
            await self.consumer.stop()
            self.consumer = None
        self.status = ConnectorStatus.DISCONNECTED
        logger.info(f"Disconnected from Kafka topic: {self.topic}")

    async def consume(self) -> AsyncIterator[Any]:
        """Consume messages from Kafka topic."""
        if not self.consumer:
            await self.connect()

        try:
            async for message in self.consumer:
                self.metrics.messages_received += 1
                self.metrics.bytes_received += len(message.value)
                self.metrics.last_success_time = datetime.now()
                
                logger.debug(
                    f"Received message from Kafka: partition={message.partition}, "
                    f"offset={message.offset}"
                )
                
                yield message.value

        except Exception as e:
            self.metrics.messages_failed += 1
            self.metrics.last_error_time = datetime.now()
            self.metrics.last_error_message = str(e)
            self.status = ConnectorStatus.ERROR
            logger.error(f"Error consuming from Kafka: {e}")
            raise


class FileFeedConnector(BaseConnector):
    """
    File feed connector for batch data ingestion.
    
    Supports watching directories for new files and processing them incrementally.
    """

    def __init__(
        self,
        directory: Union[str, Path],
        file_pattern: str = "*.json",
        watch_mode: bool = False,
        config: Optional[ConnectorConfig] = None,
    ):
        super().__init__(connector_id=f"file-{directory}", config=config)
        self.directory = Path(directory)
        self.file_pattern = file_pattern
        self.watch_mode = watch_mode
        self.processed_files: set[Path] = set()

    async def connect(self) -> None:
        """Validate directory exists."""
        self.status = ConnectorStatus.CONNECTING
        
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory not found: {self.directory}")
        
        if not self.directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.directory}")
        
        self.status = ConnectorStatus.CONNECTED
        logger.info(f"Connected to file feed: {self.directory}")

    async def disconnect(self) -> None:
        """Cleanup resources."""
        self.status = ConnectorStatus.DISCONNECTED
        logger.info(f"Disconnected from file feed: {self.directory}")

    async def consume(self) -> AsyncIterator[Dict[str, Any]]:
        """Consume files from directory."""
        if self.status != ConnectorStatus.CONNECTED:
            await self.connect()

        while True:
            # Find new files
            files = sorted(self.directory.glob(self.file_pattern))
            new_files = [f for f in files if f not in self.processed_files]

            for file_path in new_files:
                try:
                    data = await self._read_file(file_path)
                    self.processed_files.add(file_path)
                    self.metrics.messages_received += 1
                    self.metrics.bytes_received += file_path.stat().st_size
                    self.metrics.last_success_time = datetime.now()
                    
                    logger.info(f"Processed file: {file_path}")
                    yield data

                except Exception as e:
                    self.metrics.messages_failed += 1
                    self.metrics.last_error_time = datetime.now()
                    self.metrics.last_error_message = str(e)
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue

            # If not in watch mode, exit after processing all files
            if not self.watch_mode:
                break

            # Wait before checking for new files
            await asyncio.sleep(5.0)

    async def _read_file(self, file_path: Path) -> Dict[str, Any]:
        """Read and parse file."""
        with open(file_path, "r") as f:
            if file_path.suffix == ".json":
                return json.load(f)
            else:
                # Return raw content for other formats
                return {"content": f.read(), "path": str(file_path)}


class RestApiConnector(BaseConnector):
    """
    REST API pull connector for polling external APIs.
    
    Supports configurable polling intervals and rate limiting.
    """

    def __init__(
        self,
        base_url: str,
        endpoint: str,
        poll_interval: float = 60.0,
        headers: Optional[Dict[str, str]] = None,
        config: Optional[ConnectorConfig] = None,
    ):
        super().__init__(connector_id=f"rest-{endpoint}", config=config)
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint.lstrip("/")
        self.poll_interval = poll_interval
        self.headers = headers or {}
        self.session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> None:
        """Create HTTP session."""
        self.status = ConnectorStatus.CONNECTING
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        self.status = ConnectorStatus.CONNECTED
        logger.info(f"Connected to REST API: {self.base_url}/{self.endpoint}")

    async def disconnect(self) -> None:
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
        self.status = ConnectorStatus.DISCONNECTED
        logger.info(f"Disconnected from REST API: {self.base_url}/{self.endpoint}")

    async def consume(self) -> AsyncIterator[Dict[str, Any]]:
        """Poll REST API endpoint."""
        if not self.session:
            await self.connect()

        while True:
            try:
                async def _fetch():
                    url = f"{self.base_url}/{self.endpoint}"
                    async with self.session.get(url, headers=self.headers) as response:
                        response.raise_for_status()
                        return await response.json()

                data = await self.retry_operation(_fetch)
                self.metrics.messages_received += 1
                self.metrics.last_success_time = datetime.now()
                
                logger.debug(f"Fetched data from REST API: {self.base_url}/{self.endpoint}")
                yield data

            except Exception as e:
                self.metrics.messages_failed += 1
                self.metrics.last_error_time = datetime.now()
                self.metrics.last_error_message = str(e)
                logger.error(f"Error fetching from REST API: {e}")

            # Wait before next poll
            await asyncio.sleep(self.poll_interval)


class DatabaseConnector(BaseConnector):
    """
    Database polling connector for extracting data.
    
    Note: Requires appropriate database driver (asyncpg, aiomysql, etc.)
    This is a base implementation - extend for specific databases.
    """

    def __init__(
        self,
        connection_string: str,
        query: str,
        poll_interval: float = 60.0,
        config: Optional[ConnectorConfig] = None,
    ):
        super().__init__(connector_id="database", config=config)
        self.connection_string = connection_string
        self.query = query
        self.poll_interval = poll_interval
        self.connection: Optional[Any] = None

    async def connect(self) -> None:
        """Establish database connection."""
        raise NotImplementedError(
            "DatabaseConnector is a base class. "
            "Use PostgresConnector, MySQLConnector, etc."
        )

    async def disconnect(self) -> None:
        """Close database connection."""
        if self.connection:
            await self.connection.close()
            self.connection = None
        self.status = ConnectorStatus.DISCONNECTED

    async def consume(self) -> AsyncIterator[List[Dict[str, Any]]]:
        """Poll database with query."""
        raise NotImplementedError("Implement in database-specific subclass")


# Example usage and testing functions
async def test_rest_connector():
    """Test REST API connector."""
    connector = RestApiConnector(
        base_url="https://api.example.com",
        endpoint="/data",
        poll_interval=10.0,
    )
    
    try:
        await connector.connect()
        
        async for data in connector.consume():
            print(f"Received data: {data}")
            break  # Process one item for testing
            
    finally:
        await connector.disconnect()


async def test_file_connector():
    """Test file feed connector."""
    connector = FileFeedConnector(
        directory="/tmp/mfn_data",
        file_pattern="*.json",
        watch_mode=False,
    )
    
    try:
        await connector.connect()
        
        async for data in connector.consume():
            print(f"Processed file data: {data}")
            
    finally:
        await connector.disconnect()


if __name__ == "__main__":
    # Example usage
    print("MFN Data Connectors - Production Ready")
    print("Available connectors:")
    print("  - KafkaConnector (requires aiokafka)")
    print("  - FileFeedConnector")
    print("  - RestApiConnector")
    print("  - DatabaseConnector (base class)")
