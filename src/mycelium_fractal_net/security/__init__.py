"""
Security module for MyceliumFractalNet.

Provides security utilities including:
    - Data encryption for sensitive data at rest
    - Input validation and sanitization
    - Audit logging for security-relevant operations
    - Secrets management utilities

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
]
