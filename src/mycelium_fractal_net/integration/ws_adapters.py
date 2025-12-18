"""
WebSocket adapters for streaming simulation data.

Provides adapters for generating streaming updates from simulations
and fractal feature computations.

Reference: docs/MFN_BACKLOG.md#MFN-API-STREAMING
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Dict

import numpy as np
from numpy.typing import NDArray

from ..core import ReactionDiffusionConfig, ReactionDiffusionEngine
from .logging_config import get_logger
from .service_context import ServiceContext
from .ws_schemas import (
    SimulationLiveParams,
    StreamFeaturesParams,
    WSFeatureUpdate,
    WSSimulationComplete,
    WSSimulationState,
)

logger = get_logger("ws_adapters")


async def stream_features_adapter(
    stream_id: str,
    params: StreamFeaturesParams,
    ctx: ServiceContext,
) -> AsyncIterator[WSFeatureUpdate]:
    """
    Generate stream of fractal features.

    Continuously computes and streams fractal features from a running simulation.

    Args:
        stream_id: Unique stream identifier.
        params: Stream parameters.
        ctx: Service context.

    Yields:
        WSFeatureUpdate: Feature updates.
    """
    sequence = 0
    update_interval = params.update_interval_ms / 1000.0  # Convert to seconds

    # Initialize engine with proper config
    config = ReactionDiffusionConfig(grid_size=ctx.grid_size, random_seed=ctx.seed)
    rd_engine = ReactionDiffusionEngine(config)
    rd_engine.initialize_field()

    logger.info(
        f"Starting feature stream: {stream_id}",
        extra={
            "stream_id": stream_id,
            "update_interval_ms": params.update_interval_ms,
            "compression": params.compression,
        },
    )

    try:
        while True:
            start_time = time.time()

            # Run a simulation step
            rd_engine.simulate(steps=1, turing_enabled=True)

            # Get field state
            field = rd_engine.field
            if field is not None:
                features = _compute_fractal_features(field)

                # Filter features if specified
                if params.features:
                    features = {
                        k: v for k, v in features.items() if k in params.features
                    }

                # Create update message
                update = WSFeatureUpdate(
                    stream_id=stream_id,
                    sequence=sequence,
                    features=features,
                    timestamp=time.time() * 1000,
                )

                yield update

            sequence += 1

            # Rate limiting
            elapsed = time.time() - start_time
            if elapsed < update_interval:
                await asyncio.sleep(update_interval - elapsed)

    except asyncio.CancelledError:
        logger.info(
            f"Feature stream cancelled: {stream_id}",
            extra={"stream_id": stream_id, "total_updates": sequence},
        )
        raise


async def stream_simulation_live_adapter(
    stream_id: str,
    params: SimulationLiveParams,
    ctx: ServiceContext,
) -> AsyncIterator[WSSimulationState | WSSimulationComplete]:
    """
    Generate live simulation state updates.

    Streams state-by-state updates as simulation progresses.

    Args:
        stream_id: Unique stream identifier.
        params: Simulation parameters.
        ctx: Service context.

    Yields:
        WSSimulationState or WSSimulationComplete: State updates or completion message.
    """
    # Override context seed if provided
    if params.seed != ctx.seed:
        ctx = ServiceContext(seed=params.seed, mode=ctx.mode)

    # Initialize engine with proper config
    config = ReactionDiffusionConfig(
        grid_size=params.grid_size,
        random_seed=params.seed,
        alpha=params.alpha,
    )
    rd_engine = ReactionDiffusionEngine(config)
    rd_engine.initialize_field()

    logger.info(
        f"Starting simulation stream: {stream_id}",
        extra={
            "stream_id": stream_id,
            "grid_size": params.grid_size,
            "steps": params.steps,
            "update_interval_steps": params.update_interval_steps,
        },
    )

    growth_events = 0

    try:
        for step in range(params.steps):
            # Run simulation step
            rd_engine.simulate(steps=1, turing_enabled=params.turing_enabled)

            # Count growth events (simplified - based on spike probability)
            if ctx.rng.random() < params.spike_probability:
                growth_events += 1

            # Send update at specified interval
            if (step + 1) % params.update_interval_steps == 0 or step == params.steps - 1:
                field = rd_engine.field
                if field is not None:
                    # Compute state metrics
                    state_data: Dict[str, Any] = {
                        "pot_mean_mV": float(np.mean(field) * 1000),
                        "pot_std_mV": float(np.std(field) * 1000),
                        "pot_min_mV": float(np.min(field) * 1000),
                        "pot_max_mV": float(np.max(field) * 1000),
                        "active_nodes": int(np.sum(np.abs(field) > 0.01)),
                    }

                    # Add full state if requested (can be large)
                    if params.include_full_state:
                        # Only include a compressed representation
                        state_data["field_shape"] = list(field.shape)
                        state_data["field_mean"] = float(np.mean(field))

                    metrics_data: Dict[str, float] = {
                        "growth_events": float(growth_events),
                    }

                    update = WSSimulationState(
                        stream_id=stream_id,
                        step=step + 1,
                        total_steps=params.steps,
                        state=state_data,
                        metrics=metrics_data,
                        timestamp=time.time() * 1000,
                    )

                    yield update

                # Allow other tasks to run
                await asyncio.sleep(0)

        # Send completion message
        final_field = rd_engine.field
        if final_field is not None:
            final_metrics = {
                "growth_events": float(growth_events),
                "pot_min_mV": float(np.min(final_field) * 1000),
                "pot_max_mV": float(np.max(final_field) * 1000),
                "pot_mean_mV": float(np.mean(final_field) * 1000),
                "pot_std_mV": float(np.std(final_field) * 1000),
                "fractal_dimension": _compute_fractal_dimension(final_field),
            }

            completion = WSSimulationComplete(
                stream_id=stream_id,
                final_metrics=final_metrics,
                timestamp=time.time() * 1000,
            )

            yield completion

        logger.info(
            f"Simulation stream completed: {stream_id}",
            extra={
                "stream_id": stream_id,
                "total_steps": params.steps,
                "growth_events": growth_events,
            },
        )

    except asyncio.CancelledError:
        logger.info(
            f"Simulation stream cancelled: {stream_id}",
            extra={"stream_id": stream_id, "steps_completed": step + 1},
        )
        raise


def _compute_fractal_features(field: NDArray[np.floating[Any]]) -> Dict[str, float]:
    """
    Compute fractal features from field.

    Args:
        field: 2D field array.

    Returns:
        Dictionary of fractal features.
    """
    features: Dict[str, float] = {}

    # Basic statistics
    features["pot_mean_mV"] = float(np.mean(field) * 1000)
    features["pot_std_mV"] = float(np.std(field) * 1000)
    features["pot_min_mV"] = float(np.min(field) * 1000)
    features["pot_max_mV"] = float(np.max(field) * 1000)

    # Active nodes
    active_count = int(np.sum(np.abs(field) > 0.01))
    features["active_nodes"] = float(active_count)
    features["activity_ratio"] = float(active_count / field.size)

    # Fractal dimension (simplified)
    features["fractal_dimension"] = _compute_fractal_dimension(field)

    # Energy-like measure
    features["total_energy"] = float(np.sum(field**2))

    # Spatial variance
    grad_x = np.gradient(field, axis=0)
    grad_y = np.gradient(field, axis=1)
    features["spatial_variance"] = float(np.mean(grad_x**2 + grad_y**2))

    return features


def _compute_fractal_dimension(field: NDArray[np.floating[Any]], threshold: float = 0.01) -> float:
    """
    Compute box-counting fractal dimension.

    Simplified implementation for streaming.

    Args:
        field: 2D field array.
        threshold: Threshold for binary conversion.

    Returns:
        Fractal dimension estimate.
    """
    # Binarize field
    binary = np.abs(field) > threshold

    # Box counting at a few scales
    sizes = [2, 4, 8, 16]
    counts: list[int] = []
    used_sizes: list[int] = []

    for size in sizes:
        if size > min(field.shape):
            break

        # Count boxes containing active pixels
        rows = field.shape[0] // size
        cols = field.shape[1] // size
        count = 0

        for i in range(rows):
            for j in range(cols):
                box = binary[i * size : (i + 1) * size, j * size : (j + 1) * size]
                if np.any(box):
                    count += 1

        counts.append(count)
        used_sizes.append(size)

    # Ignore scales with zero occupied boxes to avoid log(0) instability
    valid_counts = [(s, c) for s, c in zip(used_sizes, counts) if c > 0]
    if len(valid_counts) < 2:
        return 0.0

    valid_sizes, counts = zip(*valid_counts)

    # Linear regression in log-log space
    log_sizes = np.log(np.asarray(valid_sizes, dtype=float))
    log_counts = np.log(np.asarray(counts, dtype=float))

    size_span = log_sizes[-1] - log_sizes[0]
    if size_span == 0:
        return 0.0

    slope = (log_counts[-1] - log_counts[0]) / size_span
    dimension = -slope
    if not np.isfinite(dimension):
        return 0.0

    return float(np.clip(dimension, 0.0, 2.0))
