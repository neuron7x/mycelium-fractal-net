"""
Security module for MyceliumFractalNet.

Provides security utilities including:
    - Data encryption for sensitive data at rest
    - Input validation and sanitization
    - Audit logging for security-relevant operations
    - System event logging for all operations
    - Secrets management utilities
    - Security iteration configuration (6 levels)

Security Standards:
    - Follows OWASP security best practices
    - Supports GDPR compliance requirements
    - SOC 2 compatible audit logging

Usage:
    >>> from mycelium_fractal_net.security import (
    ...     encrypt_data,
    ...     decrypt_data,
    ...     validate_input,
    ...     audit_log,
    ...     log_system_event,
    ...     SecurityLevel,
    ...     get_iteration_count,
    ... )

Reference: docs/MFN_SECURITY.md
"""

from .audit import (
    AuditEvent,
    AuditLogger,
    AuditSeverity,
    audit_log,
    get_audit_logger,
)
from .event_logger import (
    EventStatus,
    EventType,
    SystemEvent,
    SystemEventLogger,
    get_event_logger,
    log_system_event,
)
from .encryption import (
    DataEncryptor,
    decrypt_data,
    encrypt_data,
    generate_key,
)
from .input_validation import (
    InputValidator,
    ValidationError,
    sanitize_string,
    validate_api_key_format,
    validate_numeric_range,
)
from .iterations import (
    ADAPTIVE_MAX_ITERATIONS,
    ADAPTIVE_MIN_ITERATIONS,
    BASE_ITERATIONS,
    ENHANCED_ITERATIONS,
    HIGH_SECURITY_ITERATIONS,
    MAXIMUM_ITERATIONS,
    QUANTUM_RESISTANT_ITERATIONS,
    SecurityIterationConfig,
    SecurityLevel,
    get_iteration_count,
    get_security_iteration_config,
    reset_security_iteration_config,
    validate_iteration_count,
)

__all__ = [
    # Encryption
    "DataEncryptor",
    "encrypt_data",
    "decrypt_data",
    "generate_key",
    # Input Validation
    "InputValidator",
    "ValidationError",
    "sanitize_string",
    "validate_api_key_format",
    "validate_numeric_range",
    # Audit
    "AuditEvent",
    "AuditLogger",
    "AuditSeverity",
    "audit_log",
    "get_audit_logger",
    # Event Logging
    "EventType",
    "EventStatus",
    "SystemEvent",
    "SystemEventLogger",
    "get_event_logger",
    "log_system_event",
    # Security Iterations (6 levels)
    "SecurityLevel",
    "SecurityIterationConfig",
    "BASE_ITERATIONS",
    "ENHANCED_ITERATIONS",
    "HIGH_SECURITY_ITERATIONS",
    "MAXIMUM_ITERATIONS",
    "ADAPTIVE_MIN_ITERATIONS",
    "ADAPTIVE_MAX_ITERATIONS",
    "QUANTUM_RESISTANT_ITERATIONS",
    "get_iteration_count",
    "get_security_iteration_config",
    "reset_security_iteration_config",
    "validate_iteration_count",
]
