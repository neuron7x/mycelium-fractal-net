"""
Tests for event logging API.

Verifies REST API endpoints for accessing event logs with access control.

Reference: Problem statement - Логування всіх подій та дій системи
"""

from __future__ import annotations

from datetime import datetime

import pytest

from mycelium_fractal_net.integration.event_api import (
    EventAccessControl,
    EventAPI,
    EventFilter,
    UserRole,
    get_event_api,
)
from mycelium_fractal_net.security.event_logger import (
    EventStatus,
    EventType,
    SystemEventLogger,
)


class TestEventAccessControl:
    """Tests for event log access control."""

    def test_admin_has_full_access(self) -> None:
        """Admins should have full access to all events."""
        has_access, filtered_user = EventAccessControl.check_access(
            user_role=UserRole.ADMIN,
            user_id="admin_user",
            requested_user_id="any_user",
        )

        assert has_access is True
        assert filtered_user == "any_user"

    def test_anonymous_has_no_access(self) -> None:
        """Anonymous users should have no access."""
        has_access, filtered_user = EventAccessControl.check_access(
            user_role=UserRole.ANONYMOUS,
            user_id="anonymous",
            requested_user_id=None,
        )

        assert has_access is False

    def test_regular_user_can_only_see_own_events(self) -> None:
        """Regular users should only see their own events."""
        has_access, filtered_user = EventAccessControl.check_access(
            user_role=UserRole.USER,
            user_id="user123",
            requested_user_id="other_user",
        )

        assert has_access is True
        # Should be forced to their own user_id
        assert filtered_user == "user123"

    def test_determine_role_admin(self) -> None:
        """Should identify admin role from API key."""
        role = EventAccessControl.determine_role(api_key="admin_key")
        assert role == UserRole.ADMIN

    def test_determine_role_user(self) -> None:
        """Should identify user role from API key."""
        role = EventAccessControl.determine_role(api_key="user_key_123")
        assert role == UserRole.USER

    def test_determine_role_anonymous(self) -> None:
        """Should identify anonymous when no key provided."""
        role = EventAccessControl.determine_role(api_key=None)
        assert role == UserRole.ANONYMOUS


class TestEventFilter:
    """Tests for event filter parameters."""

    def test_filter_creation_defaults(self) -> None:
        """Should create filter with default values."""
        filter_params = EventFilter()

        assert filter_params.date is None
        assert filter_params.event_type is None
        assert filter_params.user_id is None
        assert filter_params.status is None
        assert filter_params.limit == 100

    def test_filter_with_all_parameters(self) -> None:
        """Should create filter with all parameters."""
        filter_params = EventFilter(
            date="2024-01-15",
            event_type=EventType.COMMIT,
            user_id="user123",
            status=EventStatus.SUCCESS,
            limit=50,
        )

        assert filter_params.date == "2024-01-15"
        assert filter_params.event_type == EventType.COMMIT
        assert filter_params.user_id == "user123"
        assert filter_params.status == EventStatus.SUCCESS
        assert filter_params.limit == 50


class TestEventAPI:
    """Tests for event API functionality."""

    @pytest.fixture
    def event_api(self) -> EventAPI:
        """Create event API instance."""
        return EventAPI()

    @pytest.fixture
    def logger_with_events(self, tmp_path) -> SystemEventLogger:
        """Create logger with some test events."""
        logger = SystemEventLogger(
            storage_enabled=True,
            storage_path=str(tmp_path),
        )

        # Log some test events
        logger.log_commit(
            user_id="alice",
            description="Commit by Alice",
            files_changed=["file1.py"],
        )
        logger.log_push(
            user_id="bob",
            description="Push by Bob",
        )
        logger.log_fetch(
            user_id="alice",
            description="Fetch by Alice",
        )

        return logger

    def test_get_event_types(self, event_api: EventAPI) -> None:
        """Should return available event types."""
        response = event_api.get_event_types()

        assert len(response.event_types) > 0
        assert EventType.COMMIT.value in response.event_types
        assert EventType.PUSH.value in response.event_types
        assert "commit" in response.descriptions

    def test_get_events_as_admin(
        self, event_api: EventAPI, logger_with_events: SystemEventLogger
    ) -> None:
        """Admin should be able to get all events."""
        # Replace the event logger in the API with our test logger
        event_api.event_logger = logger_with_events

        filter_params = EventFilter()
        response = event_api.get_events(
            filter_params=filter_params,
            user_role=UserRole.ADMIN,
            user_id="admin",
        )

        assert response.total >= 3
        assert len(response.events) >= 3

    def test_get_events_as_user_filtered(
        self, event_api: EventAPI, logger_with_events: SystemEventLogger
    ) -> None:
        """Regular user should only see their own events."""
        event_api.event_logger = logger_with_events

        filter_params = EventFilter()
        response = event_api.get_events(
            filter_params=filter_params,
            user_role=UserRole.USER,
            user_id="alice",
        )

        # Alice should only see her events
        assert response.total >= 2
        assert all(e.user_id == "alice" for e in response.events)

    def test_get_events_as_anonymous_denied(self, event_api: EventAPI) -> None:
        """Anonymous user should be denied access."""
        filter_params = EventFilter()

        with pytest.raises(PermissionError, match="Access denied"):
            event_api.get_events(
                filter_params=filter_params,
                user_role=UserRole.ANONYMOUS,
                user_id="anonymous",
            )

    def test_get_events_with_type_filter(
        self, event_api: EventAPI, logger_with_events: SystemEventLogger
    ) -> None:
        """Should filter events by type."""
        event_api.event_logger = logger_with_events

        filter_params = EventFilter(event_type=EventType.COMMIT)
        response = event_api.get_events(
            filter_params=filter_params,
            user_role=UserRole.ADMIN,
            user_id="admin",
        )

        assert all(e.event_type == "commit" for e in response.events)

    def test_get_events_with_status_filter(
        self, event_api: EventAPI, logger_with_events: SystemEventLogger
    ) -> None:
        """Should filter events by status."""
        event_api.event_logger = logger_with_events

        filter_params = EventFilter(status=EventStatus.SUCCESS)
        response = event_api.get_events(
            filter_params=filter_params,
            user_role=UserRole.ADMIN,
            user_id="admin",
        )

        assert all(e.status == "success" for e in response.events)

    def test_get_events_with_limit(
        self, event_api: EventAPI, logger_with_events: SystemEventLogger
    ) -> None:
        """Should respect limit parameter."""
        event_api.event_logger = logger_with_events

        filter_params = EventFilter(limit=2)
        response = event_api.get_events(
            filter_params=filter_params,
            user_role=UserRole.ADMIN,
            user_id="admin",
        )

        assert len(response.events) <= 2

    def test_get_event_stats_as_admin(
        self, event_api: EventAPI, logger_with_events: SystemEventLogger
    ) -> None:
        """Admin should get event statistics."""
        event_api.event_logger = logger_with_events

        response = event_api.get_event_stats(
            date=None,
            user_role=UserRole.ADMIN,
            user_id="admin",
        )

        assert response.total_events >= 3
        assert "commit" in response.events_by_type
        assert "push" in response.events_by_type
        assert "success" in response.events_by_status
        assert response.unique_users >= 2

    def test_get_event_stats_as_user(
        self, event_api: EventAPI, logger_with_events: SystemEventLogger
    ) -> None:
        """Regular user should get stats for their own events only."""
        event_api.event_logger = logger_with_events

        response = event_api.get_event_stats(
            date=None,
            user_role=UserRole.USER,
            user_id="alice",
        )

        # Should only count Alice's events
        assert response.total_events >= 2
        assert response.unique_users == 1

    def test_get_event_stats_as_anonymous_denied(self, event_api: EventAPI) -> None:
        """Anonymous user should be denied access to stats."""
        with pytest.raises(PermissionError, match="Access denied"):
            event_api.get_event_stats(
                date=None,
                user_role=UserRole.ANONYMOUS,
                user_id="anonymous",
            )

    def test_get_event_api_singleton(self) -> None:
        """Should return same API instance."""
        api1 = get_event_api()
        api2 = get_event_api()

        assert api1 is api2


class TestEventResponse:
    """Tests for event response models."""

    def test_event_response_from_system_event(self) -> None:
        """Should convert SystemEvent to EventResponse."""
        from mycelium_fractal_net.integration.event_api import EventResponse
        from mycelium_fractal_net.security.event_logger import SystemEvent

        system_event = SystemEvent(
            event_type=EventType.COMMIT,
            user_id="user123",
            action_description="Test commit",
            files_changed=["file.py"],
            comments="Test comment",
        )

        response = EventResponse.from_system_event(system_event)

        assert response.event_type == "commit"
        assert response.user_id == "user123"
        assert response.action_description == "Test commit"
        assert response.files_changed == ["file.py"]
        assert response.comments == "Test comment"

    def test_event_list_response_structure(self) -> None:
        """Should have proper list response structure."""
        from mycelium_fractal_net.integration.event_api import (
            EventListResponse,
            EventResponse,
        )

        events = [
            EventResponse(
                event_type="commit",
                user_id="user1",
                action_description="Test",
                timestamp=datetime.utcnow().isoformat() + "Z",
                status="success",
            )
        ]

        response = EventListResponse(
            events=events,
            total=1,
            date="2024-01-15",
            filters_applied={"date": "2024-01-15"},
        )

        assert len(response.events) == 1
        assert response.total == 1
        assert response.date == "2024-01-15"

    def test_event_stats_response_structure(self) -> None:
        """Should have proper stats response structure."""
        from mycelium_fractal_net.integration.event_api import EventStatsResponse

        response = EventStatsResponse(
            date="2024-01-15",
            total_events=100,
            events_by_type={"commit": 50, "push": 50},
            events_by_status={"success": 95, "failure": 5},
            unique_users=10,
            avg_duration_ms=123.45,
        )

        assert response.total_events == 100
        assert response.events_by_type["commit"] == 50
        assert response.unique_users == 10
        assert response.avg_duration_ms == 123.45


class TestAccessControlRequirements:
    """Tests for access control requirements."""

    def test_admins_have_full_access_to_logs(self) -> None:
        """Requirement: Logs should be accessible to administrators."""
        has_access, _ = EventAccessControl.check_access(
            user_role=UserRole.ADMIN,
            user_id="admin",
        )
        assert has_access is True

    def test_regular_users_have_limited_access(self) -> None:
        """Requirement: Logs should have restricted access for other users."""
        has_access, filtered_user = EventAccessControl.check_access(
            user_role=UserRole.USER,
            user_id="user123",
        )
        # Has access but limited to their own events
        assert has_access is True
        assert filtered_user == "user123"

    def test_anonymous_users_cannot_access_logs(self) -> None:
        """Requirement: Unauthorized users should not access logs."""
        has_access, _ = EventAccessControl.check_access(
            user_role=UserRole.ANONYMOUS,
            user_id="anon",
        )
        assert has_access is False
