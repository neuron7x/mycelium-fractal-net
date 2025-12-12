"""
Tests for comprehensive system event logging.

Verifies event logging for all system operations including commits,
pushes, pull requests, file changes, authentication, etc.

Reference: Problem statement - Логування всіх подій та дій системи
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mycelium_fractal_net.security.event_logger import (
    EventStatus,
    EventType,
    SystemEvent,
    SystemEventLogger,
    get_event_logger,
    log_system_event,
)


class TestSystemEvent:
    """Tests for SystemEvent class."""

    def test_event_creation(self) -> None:
        """Should create event with required fields."""
        event = SystemEvent(
            event_type=EventType.COMMIT,
            user_id="user123",
            action_description="Configuration change",
        )

        assert event.event_type == EventType.COMMIT
        assert event.user_id == "user123"
        assert event.action_description == "Configuration change"
        assert event.status == EventStatus.SUCCESS

    def test_event_with_files(self) -> None:
        """Should store list of changed files."""
        files = ["config.yaml", "model.pt"]
        event = SystemEvent(
            event_type=EventType.FILE_CHANGE,
            user_id="user123",
            action_description="Modified configuration",
            files_changed=files,
        )

        assert event.files_changed == files

    def test_event_with_comments(self) -> None:
        """Should store comments."""
        comment = "Updated parameters for better performance"
        event = SystemEvent(
            event_type=EventType.COMMENT,
            user_id="admin",
            action_description="Comment added",
            comments=comment,
        )

        assert event.comments == comment

    def test_event_with_metadata(self) -> None:
        """Should store additional metadata."""
        metadata = {
            "endpoint": "/api/config",
            "method": "POST",
            "status_code": 200,
        }
        event = SystemEvent(
            event_type=EventType.API_REQUEST,
            user_id="api_user",
            action_description="API request",
            metadata=metadata,
        )

        assert event.metadata == metadata

    def test_event_to_dict(self) -> None:
        """Should convert event to dictionary."""
        event = SystemEvent(
            event_type=EventType.PUSH,
            user_id="user123",
            action_description="Data push",
            files_changed=["data.json"],
            comments="Pushed new data",
        )

        data = event.to_dict()

        assert data["system_event"] is True
        assert data["event_type"] == "push"
        assert data["user_id"] == "user123"
        assert data["action_description"] == "Data push"
        assert data["files_changed"] == ["data.json"]
        assert data["comments"] == "Pushed new data"

    def test_event_to_json(self) -> None:
        """Should serialize event to JSON."""
        event = SystemEvent(
            event_type=EventType.FETCH,
            user_id="user123",
            action_description="Data fetch",
        )

        json_str = event.to_json()
        data = json.loads(json_str)

        assert data["event_type"] == "fetch"
        assert data["user_id"] == "user123"

    def test_event_timestamp(self) -> None:
        """Should include ISO 8601 timestamp."""
        event = SystemEvent(
            event_type=EventType.MERGE,
            user_id="user123",
            action_description="Data merge",
        )

        assert event.timestamp.endswith("Z")

    def test_event_with_request_context(self) -> None:
        """Should include request context."""
        event = SystemEvent(
            event_type=EventType.API_REQUEST,
            user_id="user123",
            action_description="API request",
            request_id="req-12345",
            source_ip="192.168.1.1",
            duration_ms=123.45,
        )

        data = event.to_dict()
        assert data["request_id"] == "req-12345"
        assert data["source_ip"] == "192.168.1.1"
        assert data["duration_ms"] == 123.45


class TestSystemEventLogger:
    """Tests for SystemEventLogger class."""

    @pytest.fixture
    def temp_storage(self) -> Path:
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def logger(self, temp_storage: Path) -> SystemEventLogger:
        """Create logger with temporary storage."""
        return SystemEventLogger(
            storage_enabled=True,
            storage_path=str(temp_storage),
        )

    def test_logger_initialization(self, logger: SystemEventLogger) -> None:
        """Should initialize with default settings."""
        assert logger.storage_enabled is True

    def test_log_event(self, logger: SystemEventLogger) -> None:
        """Should log events."""
        event = logger.log_event(
            event_type=EventType.COMMIT,
            user_id="user123",
            action_description="Config change",
        )

        assert event.event_type == EventType.COMMIT
        assert event.user_id == "user123"

    def test_log_commit(self, logger: SystemEventLogger) -> None:
        """Should log commit events."""
        event = logger.log_commit(
            user_id="developer",
            description="Updated model parameters",
            files_changed=["config.yaml", "model.pt"],
        )

        assert event.event_type == EventType.COMMIT
        assert len(event.files_changed) == 2

    def test_log_push(self, logger: SystemEventLogger) -> None:
        """Should log push events."""
        event = logger.log_push(
            user_id="publisher",
            description="Published model to production",
        )

        assert event.event_type == EventType.PUSH

    def test_log_pull_request(self, logger: SystemEventLogger) -> None:
        """Should log pull request events."""
        event = logger.log_pull_request(
            user_id="requester",
            description="Request for latest model",
        )

        assert event.event_type == EventType.PULL_REQUEST

    def test_log_file_change(self, logger: SystemEventLogger) -> None:
        """Should log file change events."""
        event = logger.log_file_change(
            user_id="admin",
            file_path="/config/settings.yaml",
            action="modified",
        )

        assert event.event_type == EventType.FILE_CHANGE
        assert "/config/settings.yaml" in event.files_changed

    def test_log_clone(self, logger: SystemEventLogger) -> None:
        """Should log clone events."""
        event = logger.log_clone(
            user_id="user123",
            description="Initial data retrieval",
        )

        assert event.event_type == EventType.CLONE

    def test_log_fetch(self, logger: SystemEventLogger) -> None:
        """Should log fetch events."""
        event = logger.log_fetch(
            user_id="client",
            description="Fetched latest data",
        )

        assert event.event_type == EventType.FETCH

    def test_log_merge(self, logger: SystemEventLogger) -> None:
        """Should log merge events."""
        event = logger.log_merge(
            user_id="aggregator",
            description="Merged gradients from 5 nodes",
        )

        assert event.event_type == EventType.MERGE

    def test_log_comment(self, logger: SystemEventLogger) -> None:
        """Should log comment events."""
        event = logger.log_comment(
            user_id="reviewer",
            comment_text="Looks good, approved",
            target="PR #123",
        )

        assert event.event_type == EventType.COMMENT
        assert event.comments == "Looks good, approved"

    def test_log_authentication_success(self, logger: SystemEventLogger) -> None:
        """Should log successful authentication."""
        event = logger.log_authentication(
            user_id="user123",
            auth_type=EventType.AUTH_LOGIN,
            success=True,
        )

        assert event.event_type == EventType.AUTH_LOGIN
        assert event.status == EventStatus.SUCCESS

    def test_log_authentication_failure(self, logger: SystemEventLogger) -> None:
        """Should log failed authentication."""
        event = logger.log_authentication(
            user_id="attacker",
            auth_type=EventType.AUTH_FAILURE,
            success=False,
        )

        assert event.event_type == EventType.AUTH_FAILURE
        assert event.status == EventStatus.FAILURE

    def test_log_api_request(self, logger: SystemEventLogger) -> None:
        """Should log API requests."""
        event = logger.log_api_request(
            user_id="api_client",
            endpoint="/api/validate",
            method="POST",
        )

        assert event.event_type == EventType.API_REQUEST
        assert "POST /api/validate" in event.action_description

    def test_event_persistence(
        self, logger: SystemEventLogger, temp_storage: Path
    ) -> None:
        """Should persist events to storage."""
        logger.log_event(
            event_type=EventType.COMMIT,
            user_id="user123",
            action_description="Test commit",
        )

        # Check that file was created
        from datetime import datetime

        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        log_file = temp_storage / f"events_{date_str}.jsonl"

        assert log_file.exists()

        # Read and verify content
        with open(log_file, "r") as f:
            line = f.readline()
            data = json.loads(line)
            assert data["event_type"] == "commit"
            assert data["user_id"] == "user123"

    def test_get_events(self, logger: SystemEventLogger) -> None:
        """Should retrieve events from storage."""
        # Log some events
        logger.log_commit(
            user_id="user1",
            description="Commit 1",
            files_changed=["file1.py"],
        )
        logger.log_push(
            user_id="user2",
            description="Push 1",
        )

        # Retrieve events
        events = logger.get_events()

        assert len(events) >= 2
        assert any(e.event_type == EventType.COMMIT for e in events)
        assert any(e.event_type == EventType.PUSH for e in events)

    def test_get_events_filtered_by_type(self, logger: SystemEventLogger) -> None:
        """Should filter events by type."""
        logger.log_commit(user_id="user1", description="Commit", files_changed=[])
        logger.log_push(user_id="user2", description="Push")

        events = logger.get_events(event_type=EventType.COMMIT)

        assert len(events) >= 1
        assert all(e.event_type == EventType.COMMIT for e in events)

    def test_get_events_filtered_by_user(self, logger: SystemEventLogger) -> None:
        """Should filter events by user ID."""
        logger.log_commit(user_id="alice", description="Commit", files_changed=[])
        logger.log_push(user_id="bob", description="Push")

        events = logger.get_events(user_id="alice")

        assert len(events) >= 1
        assert all(e.user_id == "alice" for e in events)


class TestEventLogFunction:
    """Tests for log_system_event convenience function."""

    def test_log_system_event_basic(self) -> None:
        """Should log event using global logger."""
        event = log_system_event(
            event_type=EventType.API_REQUEST,
            user_id="user123",
            action_description="Test API request",
        )

        assert event.event_type == EventType.API_REQUEST
        assert event.user_id == "user123"

    def test_log_system_event_with_details(self) -> None:
        """Should log event with additional details."""
        event = log_system_event(
            event_type=EventType.MERGE,
            user_id="aggregator",
            action_description="Data aggregation",
            metadata={"nodes": 10, "success_rate": 0.95},
            duration_ms=456.78,
        )

        assert event.metadata["nodes"] == 10
        assert event.duration_ms == 456.78

    def test_get_event_logger_singleton(self) -> None:
        """Should return same logger instance."""
        logger1 = get_event_logger()
        logger2 = get_event_logger()

        assert logger1 is logger2


class TestEventTypes:
    """Tests for event type coverage."""

    def test_all_event_types_loggable(self) -> None:
        """Should be able to log all event types."""
        logger = SystemEventLogger(storage_enabled=False)

        for event_type in EventType:
            event = logger.log_event(
                event_type=event_type,
                user_id="test_user",
                action_description=f"Test {event_type.value}",
            )
            assert event.event_type == event_type

    def test_event_types_have_unique_values(self) -> None:
        """Event types should have unique string values."""
        values = [e.value for e in EventType]
        assert len(values) == len(set(values))


class TestComprehensiveCoverage:
    """Tests for comprehensive event coverage as per requirements."""

    def test_logs_commit_events(self) -> None:
        """Should log commits (requirement 1)."""
        logger = SystemEventLogger(storage_enabled=False)
        event = logger.log_commit(
            user_id="dev",
            description="Updated config",
            files_changed=["config.yaml"],
        )
        assert event.event_type == EventType.COMMIT

    def test_logs_push_events(self) -> None:
        """Should log pushes (requirement 2)."""
        logger = SystemEventLogger(storage_enabled=False)
        event = logger.log_push(user_id="dev", description="Push to prod")
        assert event.event_type == EventType.PUSH

    def test_logs_pull_request_events(self) -> None:
        """Should log pull requests (requirement 3)."""
        logger = SystemEventLogger(storage_enabled=False)
        event = logger.log_pull_request(user_id="dev", description="PR opened")
        assert event.event_type == EventType.PULL_REQUEST

    def test_logs_file_changes(self) -> None:
        """Should log file changes (requirement 4)."""
        logger = SystemEventLogger(storage_enabled=False)
        event = logger.log_file_change(
            user_id="dev",
            file_path="test.py",
            action="modified",
        )
        assert event.event_type == EventType.FILE_CHANGE
        assert "test.py" in event.files_changed

    def test_logs_clone_fetch_events(self) -> None:
        """Should log clone/fetch operations (requirement 5)."""
        logger = SystemEventLogger(storage_enabled=False)

        clone_event = logger.log_clone(user_id="user", description="Initial clone")
        assert clone_event.event_type == EventType.CLONE

        fetch_event = logger.log_fetch(user_id="user", description="Fetch update")
        assert fetch_event.event_type == EventType.FETCH

    def test_logs_merge_events(self) -> None:
        """Should log merges (requirement 6)."""
        logger = SystemEventLogger(storage_enabled=False)
        event = logger.log_merge(user_id="dev", description="Merged branches")
        assert event.event_type == EventType.MERGE

    def test_logs_comments(self) -> None:
        """Should log comments to pull requests (requirement 7)."""
        logger = SystemEventLogger(storage_enabled=False)
        event = logger.log_comment(
            user_id="reviewer",
            comment_text="LGTM",
            target="PR #42",
        )
        assert event.event_type == EventType.COMMENT
        assert event.comments == "LGTM"

    def test_logs_authentication_actions(self) -> None:
        """Should log authentication actions (requirement 8)."""
        logger = SystemEventLogger(storage_enabled=False)

        login = logger.log_authentication(
            user_id="user",
            auth_type=EventType.AUTH_LOGIN,
            success=True,
        )
        assert login.event_type == EventType.AUTH_LOGIN

        logout = logger.log_authentication(
            user_id="user",
            auth_type=EventType.AUTH_LOGOUT,
            success=True,
        )
        assert logout.event_type == EventType.AUTH_LOGOUT

        api_key = logger.log_authentication(
            user_id="user",
            auth_type=EventType.AUTH_API_KEY,
            success=True,
        )
        assert api_key.event_type == EventType.AUTH_API_KEY

    def test_event_includes_timestamp(self) -> None:
        """Should include timestamp for each event."""
        logger = SystemEventLogger(storage_enabled=False)
        event = logger.log_event(
            event_type=EventType.COMMIT,
            user_id="user",
            action_description="Test",
        )
        assert event.timestamp
        assert isinstance(event.timestamp, str)

    def test_event_includes_user_id(self) -> None:
        """Should include user identifier."""
        logger = SystemEventLogger(storage_enabled=False)
        event = logger.log_event(
            event_type=EventType.COMMIT,
            user_id="test_user_123",
            action_description="Test",
        )
        assert event.user_id == "test_user_123"

    def test_event_includes_event_type(self) -> None:
        """Should include event type."""
        logger = SystemEventLogger(storage_enabled=False)
        event = logger.log_event(
            event_type=EventType.PUSH,
            user_id="user",
            action_description="Test",
        )
        assert event.event_type == EventType.PUSH

    def test_event_includes_description(self) -> None:
        """Should include action description."""
        logger = SystemEventLogger(storage_enabled=False)
        description = "Updated configuration settings"
        event = logger.log_event(
            event_type=EventType.COMMIT,
            user_id="user",
            action_description=description,
        )
        assert event.action_description == description

    def test_event_includes_files_changed(self) -> None:
        """Should include changed files if applicable."""
        logger = SystemEventLogger(storage_enabled=False)
        files = ["file1.py", "file2.py"]
        event = logger.log_event(
            event_type=EventType.FILE_CHANGE,
            user_id="user",
            action_description="Modified files",
            files_changed=files,
        )
        assert event.files_changed == files

    def test_event_includes_comments_if_applicable(self) -> None:
        """Should include comments if applicable."""
        logger = SystemEventLogger(storage_enabled=False)
        comment = "This needs review"
        event = logger.log_event(
            event_type=EventType.COMMENT,
            user_id="user",
            action_description="Comment added",
            comments=comment,
        )
        assert event.comments == comment
