"""
Canonical data structures for MyceliumFractalNet.

This module provides the unified domain model for MFN, consolidating all
canonical types that flow through the system. Each type is documented with
its mathematical/physical meaning and references to the relevant documentation.

Type Categories:
    - **Configuration Types**: SimulationConfig, FeatureConfig, DatasetConfig
    - **Result Types**: SimulationResult
    - **Field Types**: FieldState, FieldHistory, GridShape
    - **Feature Types**: FeatureVector, FeatureSchema
    - **Scenario Types**: ScenarioConfig, ScenarioType, DatasetRow, DatasetMeta
    - **API Types**: Request/Response models for REST API

Design Principles:
    1. Single source of truth - each concept is defined once
    2. Aligned with documentation - field names match MFN_FEATURE_SCHEMA.md
    3. Validation on construction - all types validate their invariants
    4. Interoperability - types support conversion to/from dict, array, DataFrame

Reference:
    - docs/MFN_DATA_MODEL.md — Data model documentation
    - docs/MFN_FEATURE_SCHEMA.md — Feature definitions
    - docs/MFN_DATA_PIPELINES.md — Dataset schema
    - docs/MFN_MATH_MODEL.md — Mathematical formalization
"""

# Re-export all canonical types for convenient access
from .config import (
    DatasetConfig,
    FeatureConfig,
    SimulationConfig,
    SimulationResult,
)
from .dataset import (
    DatasetMeta,
    DatasetRow,
    DatasetStats,
)
from .features import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FeatureVector,
)
from .field import (
    BoundaryCondition,
    FieldHistory,
    FieldState,
    GridShape,
)
from .scenario import (
    ScenarioConfig,
    ScenarioType,
)

__all__ = [
    # Configuration
    "SimulationConfig",
    "SimulationResult",
    "FeatureConfig",
    "DatasetConfig",
    # Field
    "FieldState",
    "FieldHistory",
    "GridShape",
    "BoundaryCondition",
    # Features
    "FeatureVector",
    "FEATURE_NAMES",
    "FEATURE_COUNT",
    # Scenario & Dataset
    "ScenarioConfig",
    "ScenarioType",
    "DatasetRow",
    "DatasetMeta",
    "DatasetStats",
]
