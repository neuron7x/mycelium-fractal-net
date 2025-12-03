"""
Unit tests for WebSocket connection manager.

Tests WebSocket manager functionality:
- Connection lifecycle
- Authentication with HMAC signatures
- Audit metrics tracking
- Backpressure handling
- Heartbeat monitoring
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from typing import Any, Dict
from unittest import mock

import pytest
import pytest_asyncio

from mycelium_fractal_net.integration.ws_manager import (
    BackpressureStrategy,
    WSConnectionManager,
    WSConnectionState,
)


@pytest.fixture
def connection_id():
    """Generate a test connection ID."""
    return "test-conn-123"


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket."""
    ws = mock.AsyncMock()
    ws.send_json = mock.AsyncMock()
    ws.receive_json = mock.AsyncMock()
    return ws


@pytest_asyncio.fixture
async def ws_manager():
    """Create WebSocket manager for testing."""
    manager = WSConnectionManager(
        backpressure_strategy=BackpressureStrategy.DROP_OLDEST,
        max_queue_size=10,
        heartbeat_interval=1.0,
        heartbeat_timeout=2.0,
    )
    yield manager
    # Cleanup
    if manager._heartbeat_task is not None:
        await manager.stop_heartbeat_monitor()


class TestWSConnectionState:
    """Tests for WSConnectionState."""

    def test_initialization(self, connection_id, mock_websocket):
        """Test connection state initialization."""
        state = WSConnectionState(
            connection_id=connection_id,
            websocket=mock_websocket,
            max_queue_size=100,
        )

        assert state.connection_id == connection_id
        assert state.websocket == mock_websocket
        assert not state.authenticated
        assert len(state.subscriptions) == 0
        assert state.packets_sent == 0
        assert state.packets_received == 0
        assert state.dropped_frames == 0
        assert len(state.stream_start_time) == 0

    def test_is_alive(self, connection_id, mock_websocket):
        """Test connection liveness check."""
        state = WSConnectionState(connection_id, mock_websocket)

        # Should be alive initially
        assert state.is_alive(timeout=60.0)

        # Should be dead after timeout
        state.last_heartbeat = time.time() - 100
        assert not state.is_alive(timeout=60.0)

    def test_update_heartbeat(self, connection_id, mock_websocket):
        """Test heartbeat timestamp update."""
        state = WSConnectionState(connection_id, mock_websocket)

        old_time = state.last_heartbeat
        time.sleep(0.01)
        state.update_heartbeat()

        assert state.last_heartbeat > old_time

    def test_add_subscription(self, connection_id, mock_websocket):
        """Test adding subscription tracks start time."""
        state = WSConnectionState(connection_id, mock_websocket)
        stream_id = "test-stream-1"

        state.add_subscription(stream_id)

        assert stream_id in state.subscriptions
        assert stream_id in state.stream_start_time
        assert isinstance(state.stream_start_time[stream_id], float)

    def test_remove_subscription(self, connection_id, mock_websocket):
        """Test removing subscription."""
        state = WSConnectionState(connection_id, mock_websocket)
        stream_id = "test-stream-1"

        state.add_subscription(stream_id)
        state.remove_subscription(stream_id)

        assert stream_id not in state.subscriptions


class TestWSConnectionManager:
    """Tests for WSConnectionManager."""

    @pytest.mark.asyncio
    async def test_connect(self, ws_manager, mock_websocket):
        """Test accepting WebSocket connection."""
        connection_id = await ws_manager.connect(mock_websocket)

        assert connection_id in ws_manager.connections
        assert ws_manager.connections[connection_id].websocket == mock_websocket
        mock_websocket.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect(self, ws_manager, mock_websocket):
        """Test disconnecting and cleanup."""
        connection_id = await ws_manager.connect(mock_websocket)

        # Add subscription
        stream_id = "test-stream"
        await ws_manager.subscribe(connection_id, stream_id, mock.Mock(), {})

        # Disconnect
        await ws_manager.disconnect(connection_id)

        assert connection_id not in ws_manager.connections
        assert stream_id not in ws_manager.stream_subscriptions

    @pytest.mark.asyncio
    async def test_authenticate_valid_api_key(self, ws_manager, mock_websocket):
        """Test authentication with valid API key."""
        with mock.patch.dict(
            "os.environ",
            {
                "MFN_API_KEY_REQUIRED": "true",
                "MFN_API_KEY": "test-key-123",
            },
            clear=False,
        ):
            from mycelium_fractal_net.integration import reset_config

            reset_config()

            connection_id = await ws_manager.connect(mock_websocket)
            timestamp = time.time() * 1000

            result = ws_manager.authenticate(
                connection_id, "test-key-123", timestamp
            )

            assert result
            assert ws_manager.connections[connection_id].authenticated

    @pytest.mark.asyncio
    async def test_authenticate_invalid_api_key(self, ws_manager, mock_websocket):
        """Test authentication with invalid API key."""
        with mock.patch.dict(
            "os.environ",
            {
                "MFN_API_KEY_REQUIRED": "true",
                "MFN_API_KEY": "valid-key",
            },
            clear=False,
        ):
            from mycelium_fractal_net.integration import reset_config

            reset_config()

            connection_id = await ws_manager.connect(mock_websocket)
            timestamp = time.time() * 1000

            result = ws_manager.authenticate(
                connection_id, "wrong-key", timestamp
            )

            assert not result
            assert not ws_manager.connections[connection_id].authenticated

    @pytest.mark.asyncio
    async def test_authenticate_expired_timestamp(self, ws_manager, mock_websocket):
        """Test authentication with expired timestamp."""
        connection_id = await ws_manager.connect(mock_websocket)

        # Timestamp 10 minutes in the past
        old_timestamp = (time.time() - 600) * 1000

        result = ws_manager.authenticate(
            connection_id, "test-key", old_timestamp
        )

        assert not result

    @pytest.mark.asyncio
    async def test_authenticate_with_valid_signature(self, ws_manager, mock_websocket):
        """Test authentication with valid HMAC signature."""
        with mock.patch.dict(
            "os.environ",
            {
                "MFN_API_KEY_REQUIRED": "true",
                "MFN_API_KEY": "secret-key",
            },
            clear=False,
        ):
            from mycelium_fractal_net.integration import reset_config

            reset_config()

            connection_id = await ws_manager.connect(mock_websocket)
            timestamp = time.time() * 1000

            # Generate valid signature
            timestamp_str = str(int(timestamp))
            signature = hmac.new(
                "secret-key".encode("utf-8"),
                timestamp_str.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()

            result = ws_manager.authenticate(
                connection_id, "secret-key", timestamp, signature
            )

            assert result
            assert ws_manager.connections[connection_id].authenticated

    @pytest.mark.asyncio
    async def test_authenticate_with_invalid_signature(self, ws_manager, mock_websocket):
        """Test authentication with invalid HMAC signature."""
        with mock.patch.dict(
            "os.environ",
            {
                "MFN_API_KEY_REQUIRED": "true",
                "MFN_API_KEY": "secret-key",
            },
            clear=False,
        ):
            from mycelium_fractal_net.integration import reset_config

            reset_config()

            connection_id = await ws_manager.connect(mock_websocket)
            timestamp = time.time() * 1000

            # Use wrong signature
            result = ws_manager.authenticate(
                connection_id, "secret-key", timestamp, "wrong-signature"
            )

            assert not result
            assert not ws_manager.connections[connection_id].authenticated

    @pytest.mark.asyncio
    async def test_subscribe_authenticated(self, ws_manager, mock_websocket):
        """Test subscription with authenticated connection."""
        connection_id = await ws_manager.connect(mock_websocket)

        # Authenticate
        ws_manager.connections[connection_id].authenticated = True

        # Subscribe
        stream_id = "test-stream"
        success = await ws_manager.subscribe(
            connection_id, stream_id, mock.Mock(), {}
        )

        assert success
        assert stream_id in ws_manager.connections[connection_id].subscriptions
        assert connection_id in ws_manager.stream_subscriptions[stream_id]

    @pytest.mark.asyncio
    async def test_subscribe_not_authenticated(self, ws_manager, mock_websocket):
        """Test subscription without authentication fails."""
        connection_id = await ws_manager.connect(mock_websocket)

        # Try to subscribe without authentication
        stream_id = "test-stream"
        success = await ws_manager.subscribe(
            connection_id, stream_id, mock.Mock(), {}
        )

        assert not success
        assert stream_id not in ws_manager.connections[connection_id].subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe(self, ws_manager, mock_websocket):
        """Test unsubscription."""
        connection_id = await ws_manager.connect(mock_websocket)
        ws_manager.connections[connection_id].authenticated = True

        # Subscribe
        stream_id = "test-stream"
        await ws_manager.subscribe(connection_id, stream_id, mock.Mock(), {})

        # Unsubscribe
        success = await ws_manager.unsubscribe(connection_id, stream_id)

        assert success
        assert stream_id not in ws_manager.connections[connection_id].subscriptions

    @pytest.mark.asyncio
    async def test_send_message(self, ws_manager, mock_websocket):
        """Test sending message to connection."""
        connection_id = await ws_manager.connect(mock_websocket)

        message = {"type": "test", "data": "hello"}
        success = await ws_manager.send_message(connection_id, message)

        assert success
        # Message should be sent immediately if queue was empty
        mock_websocket.send_json.assert_called()

    @pytest.mark.asyncio
    async def test_backpressure_drop_oldest(self, ws_manager, mock_websocket):
        """Test backpressure with drop_oldest strategy."""
        # Make send_json slow to prevent immediate flush
        mock_websocket.send_json.side_effect = lambda x: asyncio.sleep(0.01)
        
        connection_id = await ws_manager.connect(mock_websocket)
        connection = ws_manager.connections[connection_id]

        # Fill queue to max (without flushing)
        for i in range(ws_manager.max_queue_size):
            connection.message_queue.append({"seq": i})

        initial_dropped = connection.dropped_frames

        # Send one more message (should trigger backpressure)
        await ws_manager.send_message(connection_id, {"seq": "overflow"}, apply_backpressure=True)

        assert connection.dropped_frames > initial_dropped

    @pytest.mark.asyncio
    async def test_send_heartbeat(self, ws_manager, mock_websocket):
        """Test sending heartbeat."""
        connection_id = await ws_manager.connect(mock_websocket)

        success = await ws_manager.send_heartbeat(connection_id)

        assert success
        # Verify heartbeat message was sent
        calls = mock_websocket.send_json.call_args_list
        assert len(calls) > 0
        last_call = calls[-1][0][0]
        assert last_call["type"] == "heartbeat"

    @pytest.mark.asyncio
    async def test_handle_pong(self, ws_manager, mock_websocket):
        """Test handling pong response."""
        connection_id = await ws_manager.connect(mock_websocket)
        connection = ws_manager.connections[connection_id]

        old_heartbeat = connection.last_heartbeat
        time.sleep(0.01)

        await ws_manager.handle_pong(connection_id)

        assert connection.last_heartbeat > old_heartbeat

    @pytest.mark.asyncio
    async def test_get_stats(self, ws_manager, mock_websocket):
        """Test getting connection statistics."""
        connection_id = await ws_manager.connect(mock_websocket)
        ws_manager.connections[connection_id].authenticated = True

        await ws_manager.subscribe(connection_id, "stream-1", mock.Mock(), {})
        await ws_manager.subscribe(connection_id, "stream-2", mock.Mock(), {})

        stats = ws_manager.get_stats()

        assert stats["total_connections"] == 1
        assert stats["authenticated_connections"] == 1
        assert stats["total_streams"] == 2
        assert stats["total_subscriptions"] == 2
        assert stats["backpressure_strategy"] == BackpressureStrategy.DROP_OLDEST

    @pytest.mark.asyncio
    async def test_heartbeat_monitor_starts(self, ws_manager):
        """Test heartbeat monitor starts properly."""
        await ws_manager.start_heartbeat_monitor()

        assert ws_manager._heartbeat_task is not None
        assert not ws_manager._heartbeat_task.done()

    @pytest.mark.asyncio
    async def test_heartbeat_monitor_stops(self, ws_manager):
        """Test heartbeat monitor stops properly."""
        await ws_manager.start_heartbeat_monitor()
        await ws_manager.stop_heartbeat_monitor()

        assert ws_manager._heartbeat_task is None

    @pytest.mark.asyncio
    async def test_audit_metrics_tracking(self, ws_manager, mock_websocket):
        """Test that audit metrics are properly tracked."""
        connection_id = await ws_manager.connect(mock_websocket)
        connection = ws_manager.connections[connection_id]

        # Send messages
        for i in range(5):
            await ws_manager.send_message(connection_id, {"seq": i})

        # Force queue flush
        await ws_manager._flush_queue(connection)

        assert connection.packets_sent == 5
        assert connection.dropped_frames == 0
