from __future__ import annotations

from mycelium_fractal_net.analytics.morphology import compute_morphology_descriptor
from mycelium_fractal_net.types.features import MorphologyDescriptor
from mycelium_fractal_net.types.field import FieldSequence


def extract(sequence: FieldSequence) -> MorphologyDescriptor:
    """Canonical morphology extraction operation."""
    return compute_morphology_descriptor(sequence)
