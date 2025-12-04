"""
Tests for downstream event publishers.

Tests cover:
- Connection establishment and teardown
- Message publishing
- Batch publishing
- Retry logic and circuit breaker
- Error handling
- Metrics collection
- HMAC signature verification (webhooks)
"""

import asyncio
import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mycelium_fractal_net.integration import (
    DeliveryGuarantee,
    PublisherConfig,
    PublisherStatus,
    WebhookPublisher,
)


class TestWebhookPublisher:
    """Tests for WebhookPublisher."""

    @pytest.fixture
    def sample_message(self):
        """Sample message for testing."""
        return {
            "D_box": 1.584,
            "V_mean": -67.5,
            "f_active": 0.42,
            "timestamp": "2025-12-04T12:00:00Z",
        }

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection."""
        publisher = WebhookPublisher(
            url="https://api.example.com/webhook",
        )

        await publisher.connect()
        assert publisher.status == PublisherStatus.CONNECTED
        assert publisher.session is not None

        await publisher.disconnect()
        assert publisher.status == PublisherStatus.DISCONNECTED
        assert publisher.session is None

    @pytest.mark.asyncio
    async def test_publish_single_message(self, sample_message):
        """Test publishing single message."""
        publisher = WebhookPublisher(
            url="https://api.example.com/webhook",
        )

        await publisher.connect()

        # Mock session.post
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            publisher.session,
            "post",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)),
        ):
            await publisher._publish_single(sample_message)

            assert publisher.metrics.messages_sent == 1
            assert publisher.metrics.bytes_sent > 0
            assert publisher.metrics.last_success_time is not None

        await publisher.disconnect()

    @pytest.mark.asyncio
    async def test_batch_publishing(self, sample_message):
        """Test batch publishing with automatic flush."""
        config = PublisherConfig(
            batch_size=3,
            batch_timeout=1.0,
        )

        publisher = WebhookPublisher(
            url="https://api.example.com/webhook",
            config=config,
        )

        await publisher.connect()

        # Mock session.post
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            publisher.session,
            "post",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)),
        ):
            # Publish 3 messages to trigger batch flush
            for i in range(3):
                await publisher.publish({**sample_message, "id": i})

            # Wait for batch to process
            await asyncio.sleep(0.1)

            assert publisher.metrics.messages_sent == 3
            assert publisher.metrics.batches_sent >= 1

        await publisher.disconnect()

    @pytest.mark.asyncio
    async def test_batch_timeout_flush(self, sample_message):
        """Test batch flush after timeout."""
        config = PublisherConfig(
            batch_size=100,
            batch_timeout=0.5,
        )

        publisher = WebhookPublisher(
            url="https://api.example.com/webhook",
            config=config,
        )

        await publisher.connect()

        # Mock session.post
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            publisher.session,
            "post",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)),
        ):
            # Publish single message
            await publisher.publish(sample_message)

            # Wait for timeout
            await asyncio.sleep(1.0)

            # Verify batch was flushed
            assert publisher.metrics.messages_sent == 1

        await publisher.disconnect()

    @pytest.mark.asyncio
    async def test_hmac_signature(self, sample_message):
        """Test HMAC signature generation."""
        secret_key = "test-secret-key"
        publisher = WebhookPublisher(
            url="https://api.example.com/webhook",
            secret_key=secret_key,
        )

        # Generate signature
        payload = json.dumps(sample_message)
        signature = publisher._sign_payload(payload)

        # Verify signature format
        assert signature.startswith("sha256=")

        # Verify signature is correct
        expected_signature = hmac.new(
            secret_key.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        assert signature == f"sha256={expected_signature}"

    @pytest.mark.asyncio
    async def test_publish_with_signature(self, sample_message):
        """Test publishing with HMAC signature."""
        publisher = WebhookPublisher(
            url="https://api.example.com/webhook",
            secret_key="test-secret",
        )

        await publisher.connect()

        # Mock session.post to capture headers
        captured_headers = {}

        async def mock_post(url, data, headers):
            nonlocal captured_headers
            captured_headers = headers

            mock = MagicMock()
            mock.status = 200
            mock.text = AsyncMock(return_value="OK")
            mock.raise_for_status = MagicMock()
            return AsyncMock(__aenter__=AsyncMock(return_value=mock))()

        with patch.object(publisher.session, "post", side_effect=mock_post):
            await publisher._publish_single(sample_message)

            # Verify signature header was added
            assert "X-Webhook-Signature" in captured_headers
            assert captured_headers["X-Webhook-Signature"].startswith("sha256=")

        await publisher.disconnect()

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, sample_message):
        """Test retry logic on failures."""
        config = PublisherConfig(
            max_retries=2,
            initial_retry_delay=0.1,
        )

        publisher = WebhookPublisher(
            url="https://api.example.com/webhook",
            config=config,
        )

        await publisher.connect()

        # Mock session.post to fail then succeed
        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Temporary failure")

            mock = MagicMock()
            mock.status = 200
            mock.text = AsyncMock(return_value="OK")
            mock.raise_for_status = MagicMock()
            return AsyncMock(__aenter__=AsyncMock(return_value=mock))()

        with patch.object(publisher.session, "post", side_effect=mock_post):
            await publisher._publish_single(sample_message)

            assert publisher.metrics.retry_count >= 2
            assert publisher.metrics.messages_sent == 1

        await publisher.disconnect()

    @pytest.mark.asyncio
    async def test_circuit_breaker(self, sample_message):
        """Test circuit breaker opens after threshold."""
        config = PublisherConfig(
            max_retries=0,
            circuit_breaker_threshold=2,
        )

        publisher = WebhookPublisher(
            url="https://api.example.com/webhook",
            config=config,
        )

        await publisher.connect()

        # Record failures
        publisher.circuit_breaker.record_failure()
        publisher.circuit_breaker.record_failure()

        assert publisher.circuit_breaker.is_open

        # Attempt operation should fail
        with pytest.raises(RuntimeError, match="Circuit breaker is open"):
            await publisher.retry_operation(AsyncMock())

        await publisher.disconnect()

    @pytest.mark.asyncio
    async def test_flush_on_disconnect(self, sample_message):
        """Test pending messages are flushed on disconnect."""
        config = PublisherConfig(
            batch_size=100,
            batch_timeout=60.0,
        )

        publisher = WebhookPublisher(
            url="https://api.example.com/webhook",
            config=config,
        )

        await publisher.connect()

        # Mock session.post
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            publisher.session,
            "post",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)),
        ):
            # Add messages to batch
            await publisher.publish(sample_message)
            await publisher.publish({**sample_message, "id": 2})

            # Disconnect should flush
            await publisher.disconnect()

            # Verify messages were sent
            assert publisher.metrics.messages_sent == 2


class TestPublisherMetrics:
    """Tests for publisher metrics collection."""

    @pytest.mark.asyncio
    async def test_metrics_initialization(self):
        """Test metrics are initialized correctly."""
        publisher = WebhookPublisher(
            url="https://api.example.com/webhook",
        )

        assert publisher.metrics.messages_sent == 0
        assert publisher.metrics.messages_failed == 0
        assert publisher.metrics.bytes_sent == 0
        assert publisher.metrics.batches_sent == 0
        assert publisher.metrics.publish_errors == 0
        assert publisher.metrics.retry_count == 0
        assert publisher.metrics.last_success_time is None
        assert publisher.metrics.last_error_time is None

    @pytest.mark.asyncio
    async def test_metrics_on_success(self):
        """Test metrics are updated on successful publish."""
        publisher = WebhookPublisher(
            url="https://api.example.com/webhook",
        )

        await publisher.connect()

        # Mock successful publish
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")
        mock_response.raise_for_status = MagicMock()

        message = {"test": "data"}

        with patch.object(
            publisher.session,
            "post",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)),
        ):
            await publisher._publish_single(message)

            assert publisher.metrics.messages_sent == 1
            assert publisher.metrics.bytes_sent > 0
            assert publisher.metrics.last_success_time is not None

        await publisher.disconnect()

    @pytest.mark.asyncio
    async def test_metrics_on_failure(self):
        """Test metrics are updated on failed publish."""
        config = PublisherConfig(max_retries=0)

        publisher = WebhookPublisher(
            url="https://api.example.com/webhook",
            config=config,
        )

        await publisher.connect()

        # Mock failed publish
        async def mock_post(*args, **kwargs):
            raise Exception("Network error")

        message = {"test": "data"}

        with patch.object(publisher.session, "post", side_effect=mock_post):
            with pytest.raises(Exception):
                await publisher._publish_single(message)

            assert publisher.metrics.publish_errors == 1
            assert publisher.metrics.last_error_time is not None
            assert publisher.metrics.last_error_message == "Network error"

        await publisher.disconnect()


class TestDeliveryGuarantees:
    """Tests for delivery guarantee mechanisms."""

    @pytest.mark.asyncio
    async def test_at_least_once_guarantee(self):
        """Test at-least-once delivery guarantee."""
        config = PublisherConfig(
            delivery_guarantee=DeliveryGuarantee.AT_LEAST_ONCE,
            batch_size=2,
            max_retries=1,
        )

        publisher = WebhookPublisher(
            url="https://api.example.com/webhook",
            config=config,
        )

        await publisher.connect()

        # Verify configuration
        assert publisher.config.delivery_guarantee == DeliveryGuarantee.AT_LEAST_ONCE

        await publisher.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
