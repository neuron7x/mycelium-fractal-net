"""
Experiments submodule for MyceliumFractalNet.

Contains dataset generation and experimental utilities.

Reference: docs/MFN_DATASET_SPEC.md
"""

from .generate_dataset import (
    SweepConfig,
    generate_dataset,
    generate_parameter_configs,
)

__all__ = [
    "SweepConfig",
    "generate_dataset",
    "generate_parameter_configs",
]
