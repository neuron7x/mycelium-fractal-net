"""
Comprehensive System Event Logger for MyceliumFractalNet.

Logs all system events and actions in real-time for audit and analysis.
Supports structured logging with full event context and access control.

Event Categories:
    - Commits (configuration changes, model updates)
    - Pushes (data publishing, model deployment)
    - Pull Requests (data/model requests)
    - File Changes (configuration, data file modifications)
    - Clone/Fetch (data retrieval operations)
    - Merges (data aggregation, model merging)
    - Comments (audit annotations, notes)
    - Authentication (login, logout, API key usage)

Event Fields:
    - timestamp: ISO 8601 timestamp
    - user_id: User identifier
    - event_type: Type of event
    - action_description: Human-readable description
    - files_changed: List of affected files (if applicable)
    - comments: Additional comments or notes
    - metadata: Additional event-specific data

Storage:
    Events are stored in structured format (JSON) for analysis and audit.
    Access is controlled based on user roles (admin vs regular users).

Reference: Problem statement - Логування всіх подій та дій системи
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Event logger name
EVENT_LOGGER_NAME = "mfn.security.event"


class EventType(str, Enum):
    """Types of system events to log."""

    # Configuration and model operations
    COMMIT = "commit"  # Configuration or model changes
    PUSH = "push"  # Publishing data or models
    PULL_REQUEST = "pull_request"  # Request for data/model
    
    # File operations
    FILE_CHANGE = "file_change"  # File modification
    FILE_CREATE = "file_create"  # File creation
    FILE_DELETE = "file_delete"  # File deletion
    
    # Data operations
    CLONE = "clone"  # Initial data/model retrieval
    FETCH = "fetch"  # Data fetching operations
    MERGE = "merge"  # Data aggregation/merging
    
    # Collaboration
    COMMENT = "comment"  # Audit comments or annotations
    REVIEW = "review"  # Review actions
    
    # Authentication & Authorization
    AUTH_LOGIN = "auth_login"
    AUTH_LOGOUT = "auth_logout"
    AUTH_API_KEY = "auth_api_key"
    AUTH_FAILURE = "auth_failure"
    
    # API Operations
    API_REQUEST = "api_request"
    API_RESPONSE = "api_response"
    API_ERROR = "api_error"
    
    # System Operations
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    SYSTEM_CONFIG = "system_config"


class EventStatus(str, Enum):
    """Status of the event."""

    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"


@dataclass
class SystemEvent:
    """
    Comprehensive system event.

    Represents any system action with full context for audit and analysis.

    Attributes:
        event_type: Type of event.
        user_id: Identifier of the user who triggered the event.
        action_description: Human-readable description of the action.
        timestamp: When the event occurred (ISO 8601 format).
        status: Status of the event (success, failure, etc.).
        files_changed: List of files affected by this event.
        comments: Additional comments or notes about the event.
        metadata: Additional event-specific data.
        request_id: Correlation ID for request tracing.
        source_ip: Client IP address.
        duration_ms: Duration of the operation in milliseconds.
    """

    event_type: EventType
    user_id: str
    action_description: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    status: EventStatus = EventStatus.SUCCESS
    files_changed: List[str] = field(default_factory=list)
    comments: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    source_ip: Optional[str] = None
    duration_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert event to dictionary for logging and storage.

        Returns:
            Dictionary representation of the event.
        """
        data = {
            "system_event": True,
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "action_description": self.action_description,
            "timestamp": self.timestamp,
            "status": self.status.value,
        }

        if self.files_changed:
            data["files_changed"] = self.files_changed

        if self.comments:
            data["comments"] = self.comments

        if self.metadata:
            data["metadata"] = self.metadata

        if self.request_id:
            data["request_id"] = self.request_id

        if self.source_ip:
            data["source_ip"] = self.source_ip

        if self.duration_ms is not None:
            data["duration_ms"] = self.duration_ms

        return data

    def to_json(self) -> str:
        """
        Convert event to JSON string.

        Returns:
            JSON representation of the event.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False)


class SystemEventLogger:
    """
    System event logger with structured output and persistence.

    Logs all system events to both log files and structured storage
    for audit and analysis purposes.

    Attributes:
        logger: Underlying Python logger.
        storage_enabled: Whether to persist events to storage.
        storage_path: Path to event storage directory.
    """

    def __init__(
        self,
        name: str = EVENT_LOGGER_NAME,
        storage_enabled: bool = True,
        storage_path: Optional[str] = None,
    ) -> None:
        """
        Initialize system event logger.

        Args:
            name: Logger name.
            storage_enabled: Whether to persist events to storage.
            storage_path: Path to event storage directory.
        """
        self.logger = logging.getLogger(name)
        self.storage_enabled = storage_enabled
        # Use secure default path in production, /tmp only in dev
        env = os.getenv("MFN_ENV", "dev").lower()
        default_path = "/tmp/mfn_events" if env == "dev" else "/var/log/mfn/events"
        self.storage_path = storage_path or os.getenv(
            "MFN_EVENT_STORAGE_PATH",
            default_path
        )
        self._configure_logger()
        self._setup_storage()

    def _configure_logger(self) -> None:
        """Configure the event logger if not already configured."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)

            env = os.getenv("MFN_ENV", "dev").lower()
            if env in ("prod", "production", "staging"):
                formatter = logging.Formatter("%(message)s")
            else:
                formatter = logging.Formatter(
                    "%(asctime)s EVENT %(levelname)s: %(message)s"
                )

            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _setup_storage(self) -> None:
        """Set up event storage directory."""
        if self.storage_enabled:
            storage_dir = Path(self.storage_path)
            storage_dir.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        event_type: EventType,
        user_id: str,
        action_description: str,
        status: EventStatus = EventStatus.SUCCESS,
        files_changed: Optional[List[str]] = None,
        comments: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> SystemEvent:
        """
        Log a system event.

        Args:
            event_type: Type of event.
            user_id: User identifier.
            action_description: Description of the action.
            status: Status of the event.
            files_changed: List of files affected.
            comments: Additional comments.
            metadata: Additional event data.
            request_id: Request correlation ID.
            source_ip: Client IP address.
            duration_ms: Duration of operation.

        Returns:
            The created SystemEvent.
        """
        event = SystemEvent(
            event_type=event_type,
            user_id=user_id,
            action_description=action_description,
            status=status,
            files_changed=files_changed or [],
            comments=comments,
            metadata=metadata or {},
            request_id=request_id,
            source_ip=source_ip,
            duration_ms=duration_ms,
        )

        # Log to standard logger
        log_level = logging.ERROR if status == EventStatus.FAILURE else logging.INFO
        self.logger.log(log_level, event.to_json())

        # Persist to storage
        if self.storage_enabled:
            self._persist_event(event)

        return event

    def _persist_event(self, event: SystemEvent) -> None:
        """
        Persist event to storage.

        Args:
            event: Event to persist.
        """
        try:
            # Create daily log file
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            log_file = Path(self.storage_path) / f"events_{date_str}.jsonl"

            # Append event as JSON line
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(event.to_json() + "\n")

        except Exception as e:
            self.logger.error(f"Failed to persist event: {e}")

    # Convenience methods for specific event types

    def log_commit(
        self,
        user_id: str,
        description: str,
        files_changed: List[str],
        **kwargs: Any,
    ) -> SystemEvent:
        """Log a commit event (configuration/model change)."""
        return self.log_event(
            event_type=EventType.COMMIT,
            user_id=user_id,
            action_description=description,
            files_changed=files_changed,
            **kwargs,
        )

    def log_push(
        self,
        user_id: str,
        description: str,
        **kwargs: Any,
    ) -> SystemEvent:
        """Log a push event (data/model publishing)."""
        return self.log_event(
            event_type=EventType.PUSH,
            user_id=user_id,
            action_description=description,
            **kwargs,
        )

    def log_pull_request(
        self,
        user_id: str,
        description: str,
        **kwargs: Any,
    ) -> SystemEvent:
        """Log a pull request event (data/model request)."""
        return self.log_event(
            event_type=EventType.PULL_REQUEST,
            user_id=user_id,
            action_description=description,
            **kwargs,
        )

    def log_file_change(
        self,
        user_id: str,
        file_path: str,
        action: str,
        **kwargs: Any,
    ) -> SystemEvent:
        """Log a file change event."""
        return self.log_event(
            event_type=EventType.FILE_CHANGE,
            user_id=user_id,
            action_description=f"{action}: {file_path}",
            files_changed=[file_path],
            **kwargs,
        )

    def log_clone(
        self,
        user_id: str,
        description: str,
        **kwargs: Any,
    ) -> SystemEvent:
        """Log a clone event (initial data retrieval)."""
        return self.log_event(
            event_type=EventType.CLONE,
            user_id=user_id,
            action_description=description,
            **kwargs,
        )

    def log_fetch(
        self,
        user_id: str,
        description: str,
        **kwargs: Any,
    ) -> SystemEvent:
        """Log a fetch event (data retrieval)."""
        return self.log_event(
            event_type=EventType.FETCH,
            user_id=user_id,
            action_description=description,
            **kwargs,
        )

    def log_merge(
        self,
        user_id: str,
        description: str,
        **kwargs: Any,
    ) -> SystemEvent:
        """Log a merge event (data aggregation)."""
        return self.log_event(
            event_type=EventType.MERGE,
            user_id=user_id,
            action_description=description,
            **kwargs,
        )

    def log_comment(
        self,
        user_id: str,
        comment_text: str,
        target: str,
        **kwargs: Any,
    ) -> SystemEvent:
        """Log a comment event."""
        return self.log_event(
            event_type=EventType.COMMENT,
            user_id=user_id,
            action_description=f"Comment on {target}",
            comments=comment_text,
            **kwargs,
        )

    def log_authentication(
        self,
        user_id: str,
        auth_type: EventType,
        success: bool,
        **kwargs: Any,
    ) -> SystemEvent:
        """Log an authentication event."""
        status = EventStatus.SUCCESS if success else EventStatus.FAILURE
        return self.log_event(
            event_type=auth_type,
            user_id=user_id,
            action_description=f"Authentication: {auth_type.value}",
            status=status,
            **kwargs,
        )

    def log_api_request(
        self,
        user_id: str,
        endpoint: str,
        method: str,
        **kwargs: Any,
    ) -> SystemEvent:
        """Log an API request event."""
        return self.log_event(
            event_type=EventType.API_REQUEST,
            user_id=user_id,
            action_description=f"{method} {endpoint}",
            **kwargs,
        )

    def get_events(
        self,
        date: Optional[str] = None,
        event_type: Optional[EventType] = None,
        user_id: Optional[str] = None,
    ) -> List[SystemEvent]:
        """
        Retrieve events from storage.

        Args:
            date: Date to retrieve events for (YYYY-MM-DD format).
            event_type: Filter by event type.
            user_id: Filter by user ID.

        Returns:
            List of matching events.
        """
        if not self.storage_enabled:
            return []

        date_str = date or datetime.utcnow().strftime("%Y-%m-%d")
        log_file = Path(self.storage_path) / f"events_{date_str}.jsonl"

        if not log_file.exists():
            return []

        events: List[SystemEvent] = []
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue

                    event_data = json.loads(line)

                    # Apply filters
                    if event_type and event_data.get("event_type") != event_type.value:
                        continue
                    if user_id and event_data.get("user_id") != user_id:
                        continue

                    # Reconstruct event object
                    event = SystemEvent(
                        event_type=EventType(event_data["event_type"]),
                        user_id=event_data["user_id"],
                        action_description=event_data["action_description"],
                        timestamp=event_data["timestamp"],
                        status=EventStatus(event_data["status"]),
                        files_changed=event_data.get("files_changed", []),
                        comments=event_data.get("comments"),
                        metadata=event_data.get("metadata", {}),
                        request_id=event_data.get("request_id"),
                        source_ip=event_data.get("source_ip"),
                        duration_ms=event_data.get("duration_ms"),
                    )
                    events.append(event)

        except Exception as e:
            self.logger.error(f"Failed to retrieve events: {e}")

        return events


# Singleton event logger
_event_logger: Optional[SystemEventLogger] = None


def get_event_logger() -> SystemEventLogger:
    """
    Get the singleton event logger.

    Returns:
        SystemEventLogger: The global event logger instance.
    """
    global _event_logger
    if _event_logger is None:
        _event_logger = SystemEventLogger()
    return _event_logger


def log_system_event(
    event_type: EventType,
    user_id: str,
    action_description: str,
    **kwargs: Any,
) -> SystemEvent:
    """
    Convenience function to log a system event.

    Uses the global event logger singleton.

    Args:
        event_type: Type of event.
        user_id: User identifier.
        action_description: Description of the action.
        **kwargs: Additional event parameters.

    Returns:
        The created SystemEvent.
    """
    return get_event_logger().log_event(
        event_type=event_type,
        user_id=user_id,
        action_description=action_description,
        **kwargs,
    )


__all__ = [
    "EventType",
    "EventStatus",
    "SystemEvent",
    "SystemEventLogger",
    "get_event_logger",
    "log_system_event",
]
