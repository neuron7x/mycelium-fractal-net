from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mycelium_fractal_net.neurochem.mwc import (
    effective_gabaa_shunt,
    effective_serotonergic_gain,
    mwc_fraction,
)
from mycelium_fractal_net.neurochem.state import NeuromodulationState


def _clip01(arr: NDArray[np.float64] | float) -> NDArray[np.float64]:
    return np.clip(arr, 0.0, 1.0)


def _rate(raw_hz: float, dt_seconds: float, fallback: float) -> float:
    if raw_hz <= 0.0:
        raw_hz = fallback
    return float(np.clip(raw_hz * dt_seconds, 0.0, 1.0))


def _normalize_triplet(
    occ_rest: NDArray[np.float64],
    occ_active: NDArray[np.float64],
    occ_des: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    occ_rest = np.clip(occ_rest, 0.0, 1.0)
    occ_active = np.clip(occ_active, 0.0, 1.0)
    occ_des = np.clip(occ_des, 0.0, 1.0)
    total = occ_rest + occ_active + occ_des
    needs_norm = total > 0.0
    if np.any(needs_norm):
        occ_rest = np.where(needs_norm, occ_rest / total, 1.0)
        occ_active = np.where(needs_norm, occ_active / total, 0.0)
        occ_des = np.where(needs_norm, occ_des / total, 0.0)
    return (
        occ_rest.astype(np.float64),
        occ_active.astype(np.float64),
        occ_des.astype(np.float64),
    )


def compute_excitability_offset_v(
    state: NeuromodulationState,
    *,
    activator: NDArray[np.float64],
    baseline_activation_offset_mv: float,
    rest_offset_mv: float,
    plasticity_scale: float,
) -> NDArray[np.float64]:
    centered_activator = np.asarray(activator, dtype=np.float64) - float(np.mean(activator))
    excitability_drive = np.clip(0.5 + 2.0 * centered_activator, 0.0, 1.0)
    occupancy_bias = (
        0.60 * state.occupancy_active
        + 0.25 * state.occupancy_resting
        - 0.15 * state.occupancy_desensitized
    )
    local_offset_mv = float(baseline_activation_offset_mv) + float(
        rest_offset_mv
    ) * occupancy_bias * (0.50 + 0.50 * excitability_drive)
    if plasticity_scale > 1.0:
        local_offset_mv += (
            float(rest_offset_mv) * 0.10 * (float(plasticity_scale) - 1.0) * state.plasticity_index
        )
    local_offset_mv = np.clip(local_offset_mv, -2.0, 2.0)
    return np.asarray(local_offset_mv / 1000.0, dtype=np.float64)


def step_neuromodulation_state(
    state: NeuromodulationState,
    *,
    dt_seconds: float,
    activator: NDArray[np.float64],
    field: NDArray[np.float64],
    gabaa: dict | None,
    serotonergic: dict | None,
    observation_noise: dict | None,
) -> NeuromodulationState:
    shape = field.shape
    if state.occupancy_resting.shape != shape:
        state = NeuromodulationState.zeros(shape)

    activator = np.asarray(activator, dtype=np.float64)
    field = np.asarray(field, dtype=np.float64)

    occ_rest = state.occupancy_resting.copy()
    occ_active = state.occupancy_active.copy()
    occ_des = state.occupancy_desensitized.copy()

    field_drive = np.clip((field + 0.070) / 0.110, 0.0, 1.0)
    activity_drive = np.clip(0.5 * activator + 0.5 * field_drive, 0.0, 1.0)

    if gabaa:
        concentration = float(gabaa.get("agonist_concentration_um", 0.0))
        rest_aff = float(gabaa.get("resting_affinity_um", 0.0))
        active_aff = float(gabaa.get("active_affinity_um", rest_aff))
        ligand_rest = mwc_fraction(concentration, rest_aff)
        ligand_active = mwc_fraction(concentration, active_aff)

        bind_rate = _rate(float(gabaa.get("k_on", 0.18)), dt_seconds, 0.18)
        unbind_rate = _rate(float(gabaa.get("k_off", 0.06)), dt_seconds, 0.06)
        des_rate = _rate(float(gabaa.get("desensitization_rate_hz", 0.0)), dt_seconds, 0.02)
        rec_rate = _rate(float(gabaa.get("recovery_rate_hz", 0.0)), dt_seconds, 0.02)

        available_rest = np.clip(1.0 - occ_active - occ_des, 0.0, 1.0)
        bind_propensity = np.clip(
            bind_rate * (0.35 * ligand_rest + 0.65 * ligand_active * activity_drive),
            0.0,
            1.0,
        )
        bind_flux = available_rest * bind_propensity
        occ_rest = occ_rest - bind_flux
        occ_active = occ_active + bind_flux

        unbind_propensity = np.clip(unbind_rate * (1.0 - ligand_active * activity_drive), 0.0, 1.0)
        unbind_flux = occ_active * unbind_propensity
        occ_active = occ_active - unbind_flux
        occ_rest = occ_rest + unbind_flux

        des_propensity = np.clip(des_rate * (0.40 + 0.60 * activity_drive), 0.0, 1.0)
        des_flux = occ_active * des_propensity
        occ_active = occ_active - des_flux
        occ_des = occ_des + des_flux

        rec_propensity = np.clip(rec_rate * (1.0 - 0.50 * activity_drive), 0.0, 1.0)
        rec_flux = occ_des * rec_propensity
        occ_des = occ_des - rec_flux
        occ_rest = occ_rest + rec_flux

        occ_rest, occ_active, occ_des = _normalize_triplet(occ_rest, occ_active, occ_des)
    else:
        occ_rest = np.ones(shape, dtype=np.float64)
        occ_active = np.zeros(shape, dtype=np.float64)
        occ_des = np.zeros(shape, dtype=np.float64)

    plasticity_scale = float((serotonergic or {}).get("plasticity_scale", 1.0))
    plasticity_drive = np.clip(
        np.abs(activator - np.mean(activator)) * 2.0 * plasticity_scale, 0.0, 1.0
    )
    if serotonergic:
        plasticity_drive = _clip01(
            plasticity_drive + float(serotonergic.get("reorganization_drive", 0.0))
        )
        effective_gain = effective_serotonergic_gain(
            plasticity_drive,
            float(serotonergic.get("gain_fluidity_coeff", 0.0)),
            float(serotonergic.get("coherence_bias", 0.0)),
        )
    else:
        effective_gain = np.zeros_like(field, dtype=np.float64)

    tonic_scale = float((gabaa or {}).get("tonic_inhibition_scale", 1.0))
    effective_inhibition = effective_gabaa_shunt(
        occ_active,
        float((gabaa or {}).get("shunt_strength", 0.0)) * tonic_scale,
    )

    if observation_noise:
        target_noise = np.full(
            shape, max(0.0, float(observation_noise.get("std", 0.0))), dtype=np.float64
        )
        smoothing = float(np.clip(observation_noise.get("temporal_smoothing", 0.0), 0.0, 1.0))
        observation_noise_gain = state.observation_noise_gain * smoothing + target_noise * (
            1.0 - smoothing
        )
    else:
        observation_noise_gain = np.zeros(shape, dtype=np.float64)

    return NeuromodulationState(
        occupancy_resting=occ_rest,
        occupancy_active=occ_active,
        occupancy_desensitized=occ_des,
        effective_inhibition=np.asarray(effective_inhibition, dtype=np.float64),
        effective_gain=np.asarray(effective_gain, dtype=np.float64),
        plasticity_index=np.asarray(plasticity_drive, dtype=np.float64),
        observation_noise_gain=np.asarray(observation_noise_gain, dtype=np.float64),
    )
