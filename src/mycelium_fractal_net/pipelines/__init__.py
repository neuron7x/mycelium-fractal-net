"""
Data pipelines module for MyceliumFractalNet.

Provides scenario-based data generation with configurable presets
for scientific, feature-generation, and benchmark scenarios.
"""

from .scenarios import (
    DatasetMeta,
    ScenarioConfig,
    ScenarioType,
    get_preset_config,
    list_presets,
    run_scenario,
)

__all__ = [
    "ScenarioConfig",
    "ScenarioType",
    "DatasetMeta",
    "run_scenario",
    "get_preset_config",
    "list_presets",
]
