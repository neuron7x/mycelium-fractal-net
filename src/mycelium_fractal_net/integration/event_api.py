"""
Event Logging API for MyceliumFractalNet.

Provides REST API endpoints for accessing system event logs.
Implements role-based access control for log access.

Endpoints:
    GET /events - List events with filtering
    GET /events/{date} - Get events for specific date
    GET /events/types - Get available event types
    GET /events/stats - Get event statistics

Access Control:
    - Admins: Full access to all events
    - Regular users: Limited access to their own events
    - Anonymous: No access

Reference: Problem statement - Логування всіх подій та дій системи
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..security.event_logger import EventStatus, EventType, SystemEvent, get_event_logger


class UserRole(str, Enum):
    """User roles for access control."""

    ADMIN = "admin"
    USER = "user"
    ANONYMOUS = "anonymous"


class EventFilter(BaseModel):
    """Filter parameters for event queries."""

    date: Optional[str] = Field(
        None,
        description="Date to filter events (YYYY-MM-DD format). Defaults to today.",
    )
    event_type: Optional[EventType] = Field(
        None,
        description="Filter by event type.",
    )
    user_id: Optional[str] = Field(
        None,
        description="Filter by user ID (admin only).",
    )
    status: Optional[EventStatus] = Field(
        None,
        description="Filter by event status.",
    )
    limit: int = Field(
        100,
        ge=1,
        le=1000,
        description="Maximum number of events to return.",
    )


class EventResponse(BaseModel):
    """Response model for a single event."""

    event_type: str
    user_id: str
    action_description: str
    timestamp: str
    status: str
    files_changed: List[str] = Field(default_factory=list)
    comments: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    request_id: Optional[str] = None
    source_ip: Optional[str] = None
    duration_ms: Optional[float] = None

    @classmethod
    def from_system_event(cls, event: SystemEvent) -> EventResponse:
        """
        Create response from SystemEvent.

        Args:
            event: SystemEvent to convert.

        Returns:
            EventResponse instance.
        """
        return cls(
            event_type=event.event_type.value,
            user_id=event.user_id,
            action_description=event.action_description,
            timestamp=event.timestamp,
            status=event.status.value,
            files_changed=event.files_changed,
            comments=event.comments,
            metadata=event.metadata,
            request_id=event.request_id,
            source_ip=event.source_ip,
            duration_ms=event.duration_ms,
        )


class EventListResponse(BaseModel):
    """Response model for event list."""

    events: List[EventResponse]
    total: int
    date: str
    filters_applied: Dict[str, Any]


class EventTypesResponse(BaseModel):
    """Response model for event types."""

    event_types: List[str]
    descriptions: Dict[str, str]


class EventStatsResponse(BaseModel):
    """Response model for event statistics."""

    date: str
    total_events: int
    events_by_type: Dict[str, int]
    events_by_status: Dict[str, int]
    unique_users: int
    avg_duration_ms: Optional[float]


class EventAccessControl:
    """Access control for event logs."""

    @staticmethod
    def check_access(
        user_role: UserRole,
        user_id: str,
        requested_user_id: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if user has access to events.

        Args:
            user_role: Role of the requesting user.
            user_id: ID of the requesting user.
            requested_user_id: User ID being queried (if filtering by user).

        Returns:
            Tuple of (has_access, filtered_user_id).
            If has_access is False, request should be denied.
            filtered_user_id is the user_id to use for filtering (may be overridden).
        """
        # Admins have full access
        if user_role == UserRole.ADMIN:
            return True, requested_user_id

        # Anonymous users have no access
        if user_role == UserRole.ANONYMOUS:
            return False, None

        # Regular users can only see their own events
        if user_role == UserRole.USER:
            # Force filter to their own user_id
            return True, user_id

        return False, None

    @staticmethod
    def determine_role(api_key: Optional[str] = None) -> UserRole:
        """
        Determine user role from API key or other credentials.

        Args:
            api_key: API key from request.

        Returns:
            User role.
        """
        # This is a simplified implementation
        # In production, you would look up the role from a database

        if not api_key:
            return UserRole.ANONYMOUS

        # Check for admin key (example)
        admin_keys = ["admin_key", "mfn_admin"]  # Example admin keys
        if any(key in api_key.lower() for key in admin_keys):
            return UserRole.ADMIN

        return UserRole.USER


class EventAPI:
    """API for accessing event logs."""

    def __init__(self) -> None:
        """Initialize event API."""
        self.event_logger = get_event_logger()
        self.access_control = EventAccessControl()

    def get_events(
        self,
        filter_params: EventFilter,
        user_role: UserRole,
        user_id: str,
    ) -> EventListResponse:
        """
        Get events with filtering and access control.

        Args:
            filter_params: Filter parameters.
            user_role: Role of requesting user.
            user_id: ID of requesting user.

        Returns:
            EventListResponse with filtered events.

        Raises:
            PermissionError: If user doesn't have access.
        """
        # Check access
        has_access, filtered_user_id = self.access_control.check_access(
            user_role=user_role,
            user_id=user_id,
            requested_user_id=filter_params.user_id,
        )

        if not has_access:
            raise PermissionError("Access denied to event logs")

        # Get events from logger
        events = self.event_logger.get_events(
            date=filter_params.date,
            event_type=filter_params.event_type,
            user_id=filtered_user_id,  # Use access-controlled user_id
        )

        # Apply status filter if specified
        if filter_params.status:
            events = [e for e in events if e.status == filter_params.status]

        # Apply limit
        total = len(events)
        events = events[: filter_params.limit]

        # Convert to response format
        event_responses = [EventResponse.from_system_event(e) for e in events]

        # Build response
        filters_applied = {
            "date": filter_params.date or datetime.utcnow().strftime("%Y-%m-%d"),
            "event_type": filter_params.event_type.value if filter_params.event_type else None,
            "user_id": filtered_user_id,
            "status": filter_params.status.value if filter_params.status else None,
            "limit": filter_params.limit,
        }

        return EventListResponse(
            events=event_responses,
            total=total,
            date=filter_params.date or datetime.utcnow().strftime("%Y-%m-%d"),
            filters_applied=filters_applied,
        )

    def get_event_types(self) -> EventTypesResponse:
        """
        Get available event types.

        Returns:
            EventTypesResponse with event types and descriptions.
        """
        descriptions = {
            EventType.COMMIT.value: "Configuration or model changes",
            EventType.PUSH.value: "Data or model publishing",
            EventType.PULL_REQUEST.value: "Data or model requests",
            EventType.FILE_CHANGE.value: "File modifications",
            EventType.FILE_CREATE.value: "File creation",
            EventType.FILE_DELETE.value: "File deletion",
            EventType.CLONE.value: "Initial data/model retrieval",
            EventType.FETCH.value: "Data fetching operations",
            EventType.MERGE.value: "Data aggregation/merging",
            EventType.COMMENT.value: "Audit comments or annotations",
            EventType.REVIEW.value: "Review actions",
            EventType.AUTH_LOGIN.value: "Login events",
            EventType.AUTH_LOGOUT.value: "Logout events",
            EventType.AUTH_API_KEY.value: "API key authentication",
            EventType.AUTH_FAILURE.value: "Authentication failures",
            EventType.API_REQUEST.value: "API requests",
            EventType.API_RESPONSE.value: "API responses",
            EventType.API_ERROR.value: "API errors",
            EventType.SYSTEM_START.value: "System startup",
            EventType.SYSTEM_STOP.value: "System shutdown",
            EventType.SYSTEM_CONFIG.value: "System configuration changes",
        }

        return EventTypesResponse(
            event_types=[t.value for t in EventType],
            descriptions=descriptions,
        )

    def get_event_stats(
        self,
        date: Optional[str],
        user_role: UserRole,
        user_id: str,
    ) -> EventStatsResponse:
        """
        Get event statistics for a date.

        Args:
            date: Date to get stats for (YYYY-MM-DD).
            user_role: Role of requesting user.
            user_id: ID of requesting user.

        Returns:
            EventStatsResponse with statistics.

        Raises:
            PermissionError: If user doesn't have access.
        """
        # Check access
        has_access, filtered_user_id = self.access_control.check_access(
            user_role=user_role,
            user_id=user_id,
            requested_user_id=None,
        )

        if not has_access:
            raise PermissionError("Access denied to event logs")

        # Get all events for the date
        events = self.event_logger.get_events(
            date=date,
            user_id=filtered_user_id,  # Filter by user if not admin
        )

        # Calculate statistics
        events_by_type: Dict[str, int] = {}
        events_by_status: Dict[str, int] = {}
        unique_users = set()
        durations = []

        for event in events:
            # Count by type
            event_type_str = event.event_type.value
            events_by_type[event_type_str] = events_by_type.get(event_type_str, 0) + 1

            # Count by status
            status_str = event.status.value
            events_by_status[status_str] = events_by_status.get(status_str, 0) + 1

            # Track unique users
            unique_users.add(event.user_id)

            # Collect durations
            if event.duration_ms is not None:
                durations.append(event.duration_ms)

        # Calculate average duration
        avg_duration = sum(durations) / len(durations) if durations else None

        return EventStatsResponse(
            date=date or datetime.utcnow().strftime("%Y-%m-%d"),
            total_events=len(events),
            events_by_type=events_by_type,
            events_by_status=events_by_status,
            unique_users=len(unique_users),
            avg_duration_ms=round(avg_duration, 2) if avg_duration else None,
        )


# Singleton instance
_event_api: Optional[EventAPI] = None


def get_event_api() -> EventAPI:
    """
    Get the singleton event API instance.

    Returns:
        EventAPI: The global event API instance.
    """
    global _event_api
    if _event_api is None:
        _event_api = EventAPI()
    return _event_api


__all__ = [
    "UserRole",
    "EventFilter",
    "EventResponse",
    "EventListResponse",
    "EventTypesResponse",
    "EventStatsResponse",
    "EventAccessControl",
    "EventAPI",
    "get_event_api",
]
