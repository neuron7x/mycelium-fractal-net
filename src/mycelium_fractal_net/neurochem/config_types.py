"""Typed configuration for neuromodulation kinetics.

Replaces ``dict | None`` parameters with TypedDicts that provide
compile-time key validation and IDE autocompletion.
"""

from __future__ import annotations

from typing import TypedDict

from mycelium_fractal_net.neurochem.constants import (
    DEFAULT_DES_RATE_HZ,
    DEFAULT_K_OFF_HZ,
    DEFAULT_K_ON_HZ,
    DEFAULT_REC_RATE_HZ,
)


class GABAAKineticsConfig(TypedDict, total=False):
    """GABA-A receptor kinetics configuration.

    All fields optional — defaults from constants.py apply.
    """

    profile: str
    agonist_concentration_um: float
    resting_affinity_um: float
    active_affinity_um: float
    k_on: float
    k_off: float
    desensitization_rate_hz: float
    recovery_rate_hz: float
    shunt_strength: float
    rest_offset_mv: float
    baseline_activation_offset_mv: float
    tonic_inhibition_scale: float
    K_R: float
    c: float
    Q: float
    L: float
    binding_sites: int
    k_leak_reduction_fraction: float


class SerotonergicKineticsConfig(TypedDict, total=False):
    """Serotonergic plasticity configuration."""

    plasticity_scale: float
    reorganization_drive: float
    gain_fluidity_coeff: float
    coherence_bias: float


class ObservationNoiseConfig(TypedDict, total=False):
    """Observation noise model configuration."""

    std: float
    temporal_smoothing: float


def as_gabaa(d: dict | None) -> GABAAKineticsConfig | None:
    """Cast a dict to GABAAKineticsConfig (passthrough, for type narrowing)."""
    if d is None:
        return None
    return GABAAKineticsConfig(**d)  # type: ignore[typeddict-item]


def as_serotonergic(d: dict | None) -> SerotonergicKineticsConfig | None:
    """Cast a dict to SerotonergicKineticsConfig."""
    if d is None:
        return None
    return SerotonergicKineticsConfig(**d)  # type: ignore[typeddict-item]


def as_observation_noise(d: dict | None) -> ObservationNoiseConfig | None:
    """Cast a dict to ObservationNoiseConfig."""
    if d is None:
        return None
    return ObservationNoiseConfig(**d)  # type: ignore[typeddict-item]
