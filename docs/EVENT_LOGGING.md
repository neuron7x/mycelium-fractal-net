# System Event Logging

Comprehensive event logging system for MyceliumFractalNet that tracks all system operations in real-time.

## Overview

The event logging system provides:

- **Comprehensive Coverage**: Logs all system events including commits, pushes, pull requests, file changes, authentication, and API operations
- **Structured Format**: Events stored in JSON format for easy analysis and processing
- **Access Control**: Role-based access (admin, user, anonymous) with appropriate restrictions
- **Persistence**: Events stored in daily log files for audit and compliance
- **Real-time Logging**: All events logged immediately as they occur
- **Rich Context**: Each event includes timestamp, user ID, action description, files changed, comments, and metadata

## Event Types

The system logs the following categories of events:

### 1. Configuration and Model Operations
- **COMMIT**: Configuration or model changes
- **PUSH**: Publishing data or models
- **PULL_REQUEST**: Request for data/model

### 2. File Operations
- **FILE_CHANGE**: File modifications
- **FILE_CREATE**: File creation
- **FILE_DELETE**: File deletion

### 3. Data Operations
- **CLONE**: Initial data/model retrieval
- **FETCH**: Data fetching operations
- **MERGE**: Data aggregation/merging

### 4. Collaboration
- **COMMENT**: Audit comments or annotations
- **REVIEW**: Review actions

### 5. Authentication & Authorization
- **AUTH_LOGIN**: Login events
- **AUTH_LOGOUT**: Logout events
- **AUTH_API_KEY**: API key authentication
- **AUTH_FAILURE**: Authentication failures

### 6. API Operations
- **API_REQUEST**: API requests
- **API_RESPONSE**: API responses
- **API_ERROR**: API errors

### 7. System Operations
- **SYSTEM_START**: System startup
- **SYSTEM_STOP**: System shutdown
- **SYSTEM_CONFIG**: System configuration changes

## Event Structure

Each event contains the following information:

```json
{
  "system_event": true,
  "event_type": "commit",
  "user_id": "user123",
  "action_description": "Updated model configuration",
  "timestamp": "2024-12-12T09:20:35.394Z",
  "status": "success",
  "files_changed": ["config.yaml", "model.pt"],
  "comments": "Improved performance parameters",
  "metadata": {
    "endpoint": "/api/config",
    "method": "POST",
    "status_code": 200
  },
  "request_id": "req-12345-abcde",
  "source_ip": "192.168.1.100",
  "duration_ms": 123.45
}
```

### Required Fields
- `event_type`: Type of event (see Event Types)
- `user_id`: User identifier
- `action_description`: Human-readable description
- `timestamp`: ISO 8601 timestamp
- `status`: Event status (success, failure, pending, in_progress)

### Optional Fields
- `files_changed`: List of affected files
- `comments`: Additional comments or notes
- `metadata`: Additional event-specific data
- `request_id`: Request correlation ID
- `source_ip`: Client IP address
- `duration_ms`: Operation duration in milliseconds

## Usage

### Logging Events Programmatically

```python
from mycelium_fractal_net.security import log_system_event, EventType

# Log a commit event
event = log_system_event(
    event_type=EventType.COMMIT,
    user_id="developer",
    action_description="Updated model parameters",
    files_changed=["config.yaml", "model.pt"],
    comments="Improved convergence rate",
)

# Log an authentication event
event = log_system_event(
    event_type=EventType.AUTH_LOGIN,
    user_id="user123",
    action_description="User logged in",
    source_ip="192.168.1.100",
)

# Log an API request with metadata
event = log_system_event(
    event_type=EventType.API_REQUEST,
    user_id="api_client",
    action_description="POST /validate",
    metadata={"status_code": 200, "duration_ms": 45.2},
    duration_ms=45.2,
)
```

### Using Convenience Methods

```python
from mycelium_fractal_net.security import get_event_logger

logger = get_event_logger()

# Log specific event types
logger.log_commit(
    user_id="dev",
    description="Updated config",
    files_changed=["config.yaml"],
)

logger.log_push(
    user_id="publisher",
    description="Pushed model to production",
)

logger.log_authentication(
    user_id="user123",
    auth_type=EventType.AUTH_LOGIN,
    success=True,
)
```

### Automatic Event Logging

The `EventLoggingMiddleware` automatically logs all API requests:

```python
from fastapi import FastAPI
from mycelium_fractal_net.integration import EventLoggingMiddleware

app = FastAPI()
app.add_middleware(EventLoggingMiddleware)
```

This will automatically log:
- All API requests with method, endpoint, and duration
- Authentication attempts
- Error responses
- Request metadata

## API Endpoints

### Get Events

```http
GET /events?date=2024-12-12&event_type=commit&limit=100
```

Query Parameters:
- `date` (optional): Date filter (YYYY-MM-DD)
- `event_type` (optional): Event type filter
- `status` (optional): Status filter (success, failure)
- `limit` (optional): Max results (default: 100, max: 1000)

Response:
```json
{
  "events": [...],
  "total": 42,
  "date": "2024-12-12",
  "filters_applied": {...}
}
```

### Get Event Types

```http
GET /events/types
```

Returns list of available event types with descriptions.

### Get Event Statistics

```http
GET /events/stats?date=2024-12-12
```

Returns statistics about events including:
- Total event count
- Events by type
- Events by status
- Unique user count
- Average duration

## Access Control

### Admin Users
- Full access to all events
- Can filter by any user
- Can view all statistics

### Regular Users
- Can only view their own events
- User filter is automatically applied
- Statistics limited to their events

### Anonymous Users
- No access to event logs
- Requests return 403 Forbidden

### Determining User Role

User role is determined from the API key:
- Keys containing "admin" → Admin role
- Valid API keys → User role
- No API key → Anonymous role

## Storage

### File Structure

Events are stored in daily log files:

```
/tmp/mfn_events/
├── events_2024-12-12.jsonl
├── events_2024-12-11.jsonl
└── events_2024-12-10.jsonl
```

Each file contains one JSON event per line (JSONL format).

### Configuration

Configure storage location via environment variable:

```bash
export MFN_EVENT_STORAGE_PATH=/var/log/mfn/events
```

Default: `/tmp/mfn_events`

### Retrieval

Retrieve events for analysis:

```python
from mycelium_fractal_net.security import get_event_logger

logger = get_event_logger()

# Get today's events
events = logger.get_events()

# Get events for specific date
events = logger.get_events(date="2024-12-12")

# Filter by event type
events = logger.get_events(event_type=EventType.COMMIT)

# Filter by user
events = logger.get_events(user_id="user123")
```

## Compliance

The event logging system supports various compliance requirements:

### GDPR Compliance
- Data minimization through structured logging
- User identifiers can be hashed/anonymized
- IP addresses partially redacted
- Audit trail for data access

### SOC 2 Compliance
- Complete audit trail of all system operations
- Immutable event records
- Timestamp accuracy (ISO 8601)
- Access control enforcement
- Security event tracking

### General Audit Requirements
- Who: User identification
- What: Action description and type
- When: Timestamp
- Where: Source IP
- How: Method and endpoint
- Result: Status and duration

## Integration with Existing Audit Logging

The event logging system complements the existing audit logging:

- **Audit Logging** (`mycelium_fractal_net.security.audit`): Security-focused events with redaction
- **Event Logging** (`mycelium_fractal_net.security.event_logger`): Comprehensive system events

Both can be used together:

```python
from mycelium_fractal_net.security import audit_log, log_system_event

# Log security event with redaction
audit_log(
    action="api_key_validated",
    user_id="user123",
    resource="/validate",
)

# Log comprehensive system event
log_system_event(
    event_type=EventType.API_REQUEST,
    user_id="user123",
    action_description="POST /validate",
)
```

## Performance Considerations

### Asynchronous Logging
Events are logged synchronously but writes are fast:
- JSON serialization: ~0.1ms per event
- File append: ~1ms per event

### Storage Size
Typical event sizes:
- Minimal event: ~200 bytes
- Average event: ~500 bytes
- Large event (with metadata): ~2KB

Storage estimates:
- 1,000 events/day: ~0.5 MB/day
- 10,000 events/day: ~5 MB/day
- 100,000 events/day: ~50 MB/day

### Log Rotation
Implement log rotation for long-term storage:

```bash
# Keep last 30 days
find /var/log/mfn/events -name "events_*.jsonl" -mtime +30 -delete

# Compress old logs
find /var/log/mfn/events -name "events_*.jsonl" -mtime +7 -exec gzip {} \;
```

## Monitoring and Alerting

### Key Metrics to Monitor
- Event logging rate
- Failed authentication attempts
- API error rate
- Average request duration
- Storage usage

### Example Monitoring

```python
from mycelium_fractal_net.security import get_event_logger

logger = get_event_logger()
events = logger.get_events()

# Count failed authentications
auth_failures = [
    e for e in events 
    if e.event_type == EventType.AUTH_FAILURE
]

if len(auth_failures) > 10:
    alert("High number of authentication failures")

# Check API error rate
api_errors = [
    e for e in events
    if e.event_type == EventType.API_ERROR
]

error_rate = len(api_errors) / len(events)
if error_rate > 0.05:  # 5% error rate
    alert("High API error rate")
```

## Troubleshooting

### Events Not Being Logged

1. Check storage path exists and is writable:
```bash
ls -la $MFN_EVENT_STORAGE_PATH
```

2. Verify middleware is added:
```python
# In api.py
app.add_middleware(EventLoggingMiddleware)
```

3. Check logger initialization:
```python
from mycelium_fractal_net.security import get_event_logger
logger = get_event_logger()
print(f"Storage enabled: {logger.storage_enabled}")
print(f"Storage path: {logger.storage_path}")
```

### Cannot Access Event Logs

1. Verify user has appropriate role:
```python
from mycelium_fractal_net.integration.event_api import EventAccessControl
role = EventAccessControl.determine_role(api_key="your_key")
print(f"Role: {role}")
```

2. Check API key is being sent:
```bash
curl -H "X-API-Key: your_key" http://localhost:8000/events
```

3. Check access control logs for denial reasons

### High Storage Usage

1. Implement log rotation (see Performance Considerations)
2. Reduce event retention period
3. Consider external log aggregation (ELK, Splunk, etc.)

## Best Practices

1. **Use Appropriate Event Types**: Choose the most specific event type
2. **Include Context**: Add relevant metadata for debugging
3. **Protect Sensitive Data**: Don't log passwords, keys, or PII
4. **Regular Review**: Monitor events for anomalies
5. **Retention Policy**: Define how long to keep logs
6. **Access Control**: Restrict log access to authorized users
7. **Backup**: Regularly backup event logs for compliance

## Future Enhancements

Potential improvements:
- Database backend for faster queries
- Full-text search capabilities
- Real-time event streaming (WebSocket)
- Event correlation and pattern detection
- Automated anomaly detection
- Integration with SIEM systems
- Event replay capabilities

## Reference

- Problem Statement: Логування всіх подій та дій системи
- Source: `src/mycelium_fractal_net/security/event_logger.py`
- API: `src/mycelium_fractal_net/integration/event_api.py`
- Middleware: `src/mycelium_fractal_net/integration/event_middleware.py`
- Tests: `tests/security/test_event_logger.py`
