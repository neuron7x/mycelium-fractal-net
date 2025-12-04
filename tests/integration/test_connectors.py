"""
Tests for external data connectors.

Tests cover:
- Connection establishment and teardown
- Message consumption
- Retry logic and circuit breaker
- Error handling
- Metrics collection
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mycelium_fractal_net.integration import (
    ConnectorConfig,
    ConnectorStatus,
    FileFeedConnector,
    RestApiConnector,
    RetryStrategy,
)


class TestFileFeedConnector:
    """Tests for FileFeedConnector."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        return {
            "simulation_params": {
                "grid_size": 64,
                "steps": 100,
                "alpha": 0.18,
            },
            "timestamp": "2025-12-04T12:00:00Z",
        }

    @pytest.mark.asyncio
    async def test_connect_success(self, temp_dir):
        """Test successful connection to directory."""
        connector = FileFeedConnector(
            directory=temp_dir,
            file_pattern="*.json",
        )

        await connector.connect()
        assert connector.status == ConnectorStatus.CONNECTED

        await connector.disconnect()
        assert connector.status == ConnectorStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_connect_nonexistent_directory(self):
        """Test connection to non-existent directory."""
        connector = FileFeedConnector(
            directory="/nonexistent/directory",
            file_pattern="*.json",
        )

        with pytest.raises(FileNotFoundError):
            await connector.connect()

    @pytest.mark.asyncio
    async def test_consume_files(self, temp_dir, sample_data):
        """Test consuming files from directory."""
        # Create test files
        file1 = temp_dir / "test1.json"
        file2 = temp_dir / "test2.json"

        with open(file1, "w") as f:
            json.dump(sample_data, f)

        with open(file2, "w") as f:
            json.dump({**sample_data, "file": 2}, f)

        # Create connector
        connector = FileFeedConnector(
            directory=temp_dir,
            file_pattern="*.json",
            watch_mode=False,
        )

        await connector.connect()

        # Consume files
        messages = []
        async for message in connector.consume():
            messages.append(message)

        await connector.disconnect()

        # Verify
        assert len(messages) == 2
        assert connector.metrics.messages_received == 2
        assert connector.metrics.bytes_received > 0

    @pytest.mark.asyncio
    async def test_watch_mode(self, temp_dir, sample_data):
        """Test watch mode for continuous monitoring."""
        connector = FileFeedConnector(
            directory=temp_dir,
            file_pattern="*.json",
            watch_mode=True,
        )

        await connector.connect()

        # Start consuming in background
        consume_task = asyncio.create_task(self._consume_first(connector))

        # Wait a bit
        await asyncio.sleep(0.5)

        # Create new file
        file1 = temp_dir / "test1.json"
        with open(file1, "w") as f:
            json.dump(sample_data, f)

        # Wait for consumption
        try:
            message = await asyncio.wait_for(consume_task, timeout=10.0)
            assert message is not None
            assert message == sample_data
        finally:
            consume_task.cancel()
            await connector.disconnect()

    @staticmethod
    async def _consume_first(connector):
        """Helper to consume first message."""
        async for message in connector.consume():
            return message

    @pytest.mark.asyncio
    async def test_metrics_collection(self, temp_dir, sample_data):
        """Test metrics are collected correctly."""
        # Create test file
        file1 = temp_dir / "test1.json"
        with open(file1, "w") as f:
            json.dump(sample_data, f)

        connector = FileFeedConnector(
            directory=temp_dir,
            file_pattern="*.json",
            watch_mode=False,
        )

        await connector.connect()

        async for _ in connector.consume():
            break

        await connector.disconnect()

        # Verify metrics
        assert connector.metrics.messages_received == 1
        assert connector.metrics.bytes_received > 0
        assert connector.metrics.last_success_time is not None


class TestRestApiConnector:
    """Tests for RestApiConnector."""

    @pytest.fixture
    def mock_response(self):
        """Mock HTTP response."""
        mock = MagicMock()
        mock.status = 200
        mock.json = AsyncMock(return_value={"data": "test"})
        mock.raise_for_status = MagicMock()
        return mock

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection."""
        connector = RestApiConnector(
            base_url="https://api.example.com",
            endpoint="/data",
            poll_interval=10.0,
        )

        await connector.connect()
        assert connector.status == ConnectorStatus.CONNECTED
        assert connector.session is not None

        await connector.disconnect()
        assert connector.status == ConnectorStatus.DISCONNECTED
        assert connector.session is None

    @pytest.mark.asyncio
    async def test_consume_success(self, mock_response):
        """Test successful data consumption."""
        connector = RestApiConnector(
            base_url="https://api.example.com",
            endpoint="/data",
            poll_interval=1.0,
        )

        await connector.connect()

        # Mock session.get
        with patch.object(
            connector.session,
            "get",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)),
        ):
            # Consume one message
            consumed = False
            async for message in connector.consume():
                assert message == {"data": "test"}
                consumed = True
                break

            assert consumed
            assert connector.metrics.messages_received == 1

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry logic on failures."""
        config = ConnectorConfig(
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            max_retries=2,
            initial_retry_delay=0.1,
        )

        connector = RestApiConnector(
            base_url="https://api.example.com",
            endpoint="/data",
            poll_interval=1.0,
            config=config,
        )

        await connector.connect()

        # Mock session.get to fail then succeed
        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Temporary failure")

            mock = MagicMock()
            mock.json = AsyncMock(return_value={"data": "success"})
            mock.raise_for_status = MagicMock()
            return AsyncMock(__aenter__=AsyncMock(return_value=mock))()

        with patch.object(connector.session, "get", side_effect=mock_get):
            async for message in connector.consume():
                assert message == {"data": "success"}
                assert connector.metrics.retry_count >= 2
                break

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_circuit_breaker(self):
        """Test circuit breaker opens after threshold."""
        config = ConnectorConfig(
            max_retries=0,
            circuit_breaker_threshold=2,
        )

        connector = RestApiConnector(
            base_url="https://api.example.com",
            endpoint="/data",
            poll_interval=1.0,
            config=config,
        )

        await connector.connect()

        # Record failures
        connector.circuit_breaker.record_failure()
        connector.circuit_breaker.record_failure()

        assert connector.circuit_breaker.is_open

        # Attempt operation should fail
        with pytest.raises(RuntimeError, match="Circuit breaker is open"):
            await connector.retry_operation(AsyncMock())

        await connector.disconnect()


class TestConnectorRetryStrategy:
    """Tests for retry strategies."""

    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        config = ConnectorConfig(
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            initial_retry_delay=1.0,
            backoff_multiplier=2.0,
        )

        connector = FileFeedConnector(
            directory="/tmp",
            config=config,
        )

        # Calculate delays
        delay1 = connector._calculate_retry_delay(1)
        delay2 = connector._calculate_retry_delay(2)
        delay3 = connector._calculate_retry_delay(3)

        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 4.0

    @pytest.mark.asyncio
    async def test_linear_backoff(self):
        """Test linear backoff calculation."""
        config = ConnectorConfig(
            retry_strategy=RetryStrategy.LINEAR_BACKOFF,
            initial_retry_delay=1.0,
        )

        connector = FileFeedConnector(
            directory="/tmp",
            config=config,
        )

        # Calculate delays
        delay1 = connector._calculate_retry_delay(1)
        delay2 = connector._calculate_retry_delay(2)
        delay3 = connector._calculate_retry_delay(3)

        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 3.0

    @pytest.mark.asyncio
    async def test_no_retry(self):
        """Test no retry strategy."""
        config = ConnectorConfig(
            retry_strategy=RetryStrategy.NO_RETRY,
        )

        connector = FileFeedConnector(
            directory="/tmp",
            config=config,
        )

        delay = connector._calculate_retry_delay(5)
        assert delay == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
