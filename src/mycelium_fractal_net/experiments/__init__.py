"""
Experiments submodule for MyceliumFractalNet.

Contains dataset generation and experimental utilities.

Reference: docs/MFN_DATASET_SPEC.md
"""

from .generate_dataset import (
    ConfigSampler,
    generate_dataset,
    to_record,
)

__all__ = [
    "ConfigSampler",
    "generate_dataset",
    "to_record",
]
