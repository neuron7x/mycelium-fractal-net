"""
Experiments submodule for MyceliumFractalNet.

Contains dataset generation and experimental utilities.

Reference: docs/MFN_DATASET_SPEC.md
"""

from .generate_dataset import (
    ConfigSampler,
    SweepConfig,
    generate_dataset,
    generate_parameter_configs,
    to_record,
)

__all__ = [
    "ConfigSampler",
    "SweepConfig",
    "generate_dataset",
    "generate_parameter_configs",
    "to_record",
]
