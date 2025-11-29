#!/usr/bin/env python
"""
Simple example: Running a single MFN simulation and extracting features.

This example demonstrates the core workflow:
1. Create a simulation configuration
2. Run the simulation
3. Extract fractal features
4. Display results

Reference: docs/MFN_INTEGRATION_SPEC.md
"""
from __future__ import annotations

import numpy as np

from analytics import compute_features
from mycelium_fractal_net import (
    SimulationConfig,
    compute_nernst_potential,
    run_mycelium_simulation_with_history,
)


def main() -> None:
    """Run simple simulation example."""
    print("=" * 60)
    print("MyceliumFractalNet Simple Simulation Example")
    print("=" * 60)

    # Step 1: Compute reference Nernst potential
    print("\n1. Computing Nernst potential for K⁺...")
    e_k = compute_nernst_potential(
        z_valence=1,
        concentration_out_molar=5e-3,   # 5 mM extracellular
        concentration_in_molar=140e-3,  # 140 mM intracellular
        temperature_k=310.0,            # 37°C
    )
    print(f"   E_K = {e_k * 1000:.2f} mV (expected ≈ -89 mV)")

    # Step 2: Configure and run simulation
    print("\n2. Running mycelium field simulation...")
    config = SimulationConfig(
        grid_size=64,
        steps=64,
        seed=42,
        turing_enabled=True,
        alpha=0.18,
    )

    result = run_mycelium_simulation_with_history(config)

    print(f"   Grid size: {result.grid_size}x{result.grid_size}")
    print(f"   Growth events: {result.growth_events}")
    print(f"   Field range: [{result.field.min()*1000:.2f}, {result.field.max()*1000:.2f}] mV")

    # Step 3: Extract features
    print("\n3. Extracting fractal features...")
    assert result.history is not None
    features = compute_features(result.history)

    print(f"   Fractal dimension (D_box): {features.D_box:.4f}")
    print(f"   Dimension R²: {features.D_r2:.4f}")
    print(f"   Mean potential (V_mean): {features.V_mean:.2f} mV")
    print(f"   Active fraction (f_active): {features.f_active:.4f}")
    print(f"   Clusters at -60mV: {features.N_clusters_low}")

    # Step 4: Validate features
    print("\n4. Validating feature ranges...")
    arr = features.to_array()
    assert not np.any(np.isnan(arr)), "NaN in features!"
    assert not np.any(np.isinf(arr)), "Inf in features!"
    assert 0.0 <= features.D_box <= 2.5, "D_box out of range"
    assert 0.0 <= features.f_active <= 1.0, "f_active out of range"
    print("   ✓ All features within expected ranges")

    # Step 5: Create dataset record
    print("\n5. Creating dataset record...")
    record = {
        "sim_id": 0,
        "random_seed": config.seed,
        "grid_size": config.grid_size,
        "steps": config.steps,
        "alpha": config.alpha,
        "turing_enabled": config.turing_enabled,
        "growth_events": result.growth_events,
        **features.to_dict(),
    }
    print(f"   Record has {len(record)} fields")
    print("   Config fields: sim_id, random_seed, grid_size, steps, alpha, turing_enabled")
    print(f"   Feature fields: {len(features.to_dict())} features")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Nernst E_K: {e_k*1000:.2f} mV")
    print(f"Simulation: {config.grid_size}x{config.grid_size} grid, {config.steps} steps")
    print(f"Fractal D: {features.D_box:.4f} (biological range: 1.4-1.9)")
    print(f"Active cells: {features.f_active*100:.1f}%")
    print("Pipeline: Config → Simulation → Features → Record ✓")


if __name__ == "__main__":
    main()
