"""
Integration layer for MyceliumFractalNet.

Provides unified schemas, service context, and adapters for consistent
operation across CLI, HTTP API, and experiment entry points.

Components:
    - schemas: Pydantic models for request/response validation
    - service_context: Unified context with config, RNG, and engine handles
    - adapters: Thin bridge between integration layer and numerical core
    - api_config: Configuration management for API features
    - crypto_config: Configuration management for cryptographic features
    - crypto_adapters: Adapters for cryptographic API endpoints
    - auth: API key authentication middleware
    - rate_limiter: Rate limiting middleware
    - metrics: Prometheus metrics collection
    - logging_config: Structured JSON logging
    - data_integrations: 77 data integrations for iteration optimization
    - ws_schemas: WebSocket message schemas for streaming
    - ws_manager: WebSocket connection manager
    - ws_adapters: Adapters for streaming simulation data
    - connectors: Upstream data connectors (REST, File, Kafka)
    - publishers: Downstream event publishers (Webhook, Kafka, File)

Usage:
    >>> from mycelium_fractal_net.integration import (
    ...     ValidateRequest,
    ...     ValidateResponse,
    ...     ServiceContext,
    ...     run_validation_adapter,
    ...     get_integration,
    ...     INTEGRATION_COUNT,
    ...     RESTConnector,
    ...     WebhookPublisher,
    ... )
    >>> ctx = ServiceContext(seed=42)
    >>> request = ValidateRequest(seed=42, epochs=1)
    >>> response = run_validation_adapter(request, ctx)

Reference: docs/ARCHITECTURE.md, docs/MFN_SYSTEM_ROLE.md, docs/MFN_INTEGRATION_GAPS.md
"""

from .adapters import (
    aggregate_gradients_adapter,
    compute_nernst_adapter,
    run_simulation_adapter,
    run_validation_adapter,
)
from .api_config import (
    APIConfig,
    AuthConfig,
    Environment,
    LoggingConfig,
    MetricsConfig,
    RateLimitConfig,
    get_api_config,
    reset_config,
)
from .auth import (
    API_KEY_HEADER,
    APIKeyMiddleware,
    require_api_key,
)
from .connectors import (
    BaseConnector,
    ConnectorConfig,
    ConnectorMetrics,
    ConnectorStatus,
    FileConnector,
    KafkaConnectorAdapter,
    RESTConnector,
)
from .connectors import (
    RetryStrategy as ConnectorRetryStrategy,
)
from .crypto_adapters import (
    CryptoAPIError,
    decrypt_data_adapter,
    encrypt_data_adapter,
    generate_keypair_adapter,
    sign_message_adapter,
    verify_signature_adapter,
)
from .crypto_config import (
    CryptoConfig,
    KeyStore,
    get_crypto_config,
    get_key_store,
    reset_crypto_config,
    reset_key_store,
)
from .data_integrations import (
    CORE_ITERATION_INTEGRATIONS,
    ENCRYPTION_OPTIMIZATION_INTEGRATIONS,
    HASH_FUNCTION_INTEGRATIONS,
    INTEGRATION_COUNT,
    KEY_DERIVATION_INTEGRATIONS,
    MEMORY_OPTIMIZATION_INTEGRATIONS,
    PARALLELIZATION_INTEGRATIONS,
    SALT_GENERATION_INTEGRATIONS,
    VALIDATION_AUDIT_INTEGRATIONS,
    DataIntegration,
    DataIntegrationConfig,
    IntegrationCategory,
    get_data_integration_config,
    get_integration,
    get_integration_categories,
    list_all_integrations,
    reset_data_integration_config,
)
from .logging_config import (
    REQUEST_ID_HEADER,
    RequestIDMiddleware,
    RequestLoggingMiddleware,
    get_logger,
    get_request_context,
    get_request_id,
    set_request_context,
    set_request_id,
    setup_logging,
)
from .metrics import (
    MetricsMiddleware,
    is_prometheus_available,
    metrics_endpoint,
)
from .publishers import (
    BasePublisher,
    FilePublisher,
    KafkaPublisherAdapter,
    PublisherConfig,
    PublisherMetrics,
    PublisherStatus,
    WebhookPublisher,
)
from .publishers import (
    RetryStrategy as PublisherRetryStrategy,
)
from .rate_limiter import (
    RateLimiter,
    RateLimitMiddleware,
)
from .schemas import (
    DecryptRequest,
    DecryptResponse,
    EncryptRequest,
    EncryptResponse,
    ErrorResponse,
    FederatedAggregateRequest,
    FederatedAggregateResponse,
    HealthResponse,
    KeypairRequest,
    KeypairResponse,
    NernstRequest,
    NernstResponse,
    SignRequest,
    SignResponse,
    SimulateRequest,
    SimulateResponse,
    ValidateRequest,
    ValidateResponse,
    VerifyRequest,
    VerifyResponse,
)
from .service_context import (
    ExecutionMode,
    ServiceContext,
    create_context_from_request,
)
from .ws_adapters import (
    stream_features_adapter,
    stream_simulation_live_adapter,
)
from .ws_manager import (
    BackpressureStrategy,
    WSConnectionManager,
    WSConnectionState,
)
from .ws_schemas import (
    SimulationLiveParams,
    StreamFeaturesParams,
    WSAuthRequest,
    WSErrorMessage,
    WSFeatureUpdate,
    WSHeartbeatRequest,
    WSInitRequest,
    WSMessage,
    WSMessageType,
    WSSimulationComplete,
    WSSimulationState,
    WSStreamType,
    WSSubscribeRequest,
    WSUnsubscribeRequest,
)

__all__ = [
    # Schemas
    "HealthResponse",
    "ValidateRequest",
    "ValidateResponse",
    "SimulateRequest",
    "SimulateResponse",
    "NernstRequest",
    "NernstResponse",
    "FederatedAggregateRequest",
    "FederatedAggregateResponse",
    "ErrorResponse",
    # Crypto Schemas
    "EncryptRequest",
    "EncryptResponse",
    "DecryptRequest",
    "DecryptResponse",
    "SignRequest",
    "SignResponse",
    "VerifyRequest",
    "VerifyResponse",
    "KeypairRequest",
    "KeypairResponse",
    # Service Context
    "ExecutionMode",
    "ServiceContext",
    "create_context_from_request",
    # Adapters
    "run_validation_adapter",
    "run_simulation_adapter",
    "compute_nernst_adapter",
    "aggregate_gradients_adapter",
    # Crypto Adapters
    "CryptoAPIError",
    "encrypt_data_adapter",
    "decrypt_data_adapter",
    "sign_message_adapter",
    "verify_signature_adapter",
    "generate_keypair_adapter",
    # API Configuration
    "Environment",
    "AuthConfig",
    "RateLimitConfig",
    "LoggingConfig",
    "MetricsConfig",
    "APIConfig",
    "get_api_config",
    "reset_config",
    # Crypto Configuration
    "CryptoConfig",
    "KeyStore",
    "get_crypto_config",
    "get_key_store",
    "reset_crypto_config",
    "reset_key_store",
    # Authentication
    "API_KEY_HEADER",
    "APIKeyMiddleware",
    "require_api_key",
    # Rate Limiting
    "RateLimiter",
    "RateLimitMiddleware",
    # Metrics
    "MetricsMiddleware",
    "metrics_endpoint",
    "is_prometheus_available",
    # Logging
    "REQUEST_ID_HEADER",
    "RequestIDMiddleware",
    "RequestLoggingMiddleware",
    "setup_logging",
    "get_logger",
    "get_request_context",
    "get_request_id",
    "set_request_context",
    "set_request_id",
    # Data Integrations (77 integrations for iteration optimization)
    "INTEGRATION_COUNT",
    "IntegrationCategory",
    "DataIntegration",
    "DataIntegrationConfig",
    "CORE_ITERATION_INTEGRATIONS",
    "KEY_DERIVATION_INTEGRATIONS",
    "ENCRYPTION_OPTIMIZATION_INTEGRATIONS",
    "HASH_FUNCTION_INTEGRATIONS",
    "SALT_GENERATION_INTEGRATIONS",
    "MEMORY_OPTIMIZATION_INTEGRATIONS",
    "PARALLELIZATION_INTEGRATIONS",
    "VALIDATION_AUDIT_INTEGRATIONS",
    "get_data_integration_config",
    "reset_data_integration_config",
    "get_integration",
    "list_all_integrations",
    "get_integration_categories",
    # WebSocket Components
    "WSMessageType",
    "WSStreamType",
    "WSMessage",
    "WSInitRequest",
    "WSAuthRequest",
    "WSSubscribeRequest",
    "WSUnsubscribeRequest",
    "WSHeartbeatRequest",
    "WSFeatureUpdate",
    "WSSimulationState",
    "WSSimulationComplete",
    "WSErrorMessage",
    "StreamFeaturesParams",
    "SimulationLiveParams",
    "WSConnectionManager",
    "WSConnectionState",
    "BackpressureStrategy",
    "stream_features_adapter",
    "stream_simulation_live_adapter",
    # Connectors (upstream data sources)
    "BaseConnector",
    "RESTConnector",
    "FileConnector",
    "KafkaConnectorAdapter",
    "ConnectorConfig",
    "ConnectorMetrics",
    "ConnectorStatus",
    "ConnectorRetryStrategy",
    # Publishers (downstream event publishing)
    "BasePublisher",
    "WebhookPublisher",
    "KafkaPublisherAdapter",
    "FilePublisher",
    "PublisherConfig",
    "PublisherMetrics",
    "PublisherStatus",
    "PublisherRetryStrategy",
]
