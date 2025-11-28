#!/usr/bin/env python
"""
Finance example: Using MyceliumFractalNet for market regime detection.

This example demonstrates how fractal dynamics can model market microstructure
and detect regime changes in financial time series.

Features used:
- Fractal dimension for market complexity analysis
- Lyapunov exponent for stability/volatility detection
- Turing morphogenesis for pattern formation
- STDP for adaptive learning from market data
"""
from __future__ import annotations

import numpy as np
import torch

from mycelium_fractal_net import (
    MyceliumFractalNet,
    estimate_fractal_dimension,
    generate_fractal_ifs,
    simulate_mycelium_field,
)


def simulate_market_data(
    rng: np.random.Generator,
    num_days: int = 252,
    volatility: float = 0.02,
) -> np.ndarray:
    """
    Simulate simple market returns with regime changes.

    Parameters
    ----------
    rng : np.random.Generator
        Random generator.
    num_days : int
        Number of trading days.
    volatility : float
        Base volatility.

    Returns
    -------
    np.ndarray
        Simulated returns.
    """
    returns = np.zeros(num_days)

    # Three regimes: low vol, high vol, trending
    regime_length = num_days // 3

    # Low volatility regime
    returns[:regime_length] = rng.normal(0.0005, volatility * 0.5, regime_length)

    # High volatility regime
    returns[regime_length : 2 * regime_length] = rng.normal(
        0, volatility * 2.0, regime_length
    )

    # Trending regime
    trend = np.linspace(0, 0.001, regime_length)
    returns[2 * regime_length :] = trend + rng.normal(0, volatility, regime_length)

    return returns


def returns_to_field(returns: np.ndarray, grid_size: int = 64) -> np.ndarray:
    """
    Convert returns to 2D field representation.

    Maps returns to membrane potentials for mycelium simulation.
    """
    # Normalize returns to [-1, 1]
    returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Create 2D embedding using rolling windows
    field = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            idx = (i * grid_size + j) % len(returns)
            # Map to membrane potential scale (-70 mV baseline)
            field[i, j] = -0.070 + returns_norm[idx] * 0.010

    return field


def detect_regime(field: np.ndarray) -> dict:
    """
    Detect market regime from field characteristics.

    Returns dict with regime indicators.
    """
    # Compute fractal dimension
    binary = field > -0.060
    fractal_dim = estimate_fractal_dimension(binary)

    # Compute statistics
    volatility = field.std() * 1000  # mV

    # Regime classification based on fractal dimension
    if fractal_dim > 1.7:
        regime = "high_complexity"
    elif fractal_dim < 1.3:
        regime = "low_complexity"
    else:
        regime = "normal"

    return {
        "fractal_dimension": fractal_dim,
        "volatility_mV": volatility,
        "regime": regime,
    }


def main() -> None:
    """Run finance example."""
    print("=" * 60)
    print("MyceliumFractalNet Finance Example")
    print("Market Regime Detection via Fractal Dynamics")
    print("=" * 60)

    # Set seed for reproducibility
    seed = 42
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Step 1: Simulate market data
    print("\n1. Simulating market data...")
    returns = simulate_market_data(rng, num_days=252)
    print(f"   Generated {len(returns)} daily returns")
    print(f"   Mean return: {returns.mean():.6f}")
    print(f"   Volatility: {returns.std():.6f}")

    # Step 2: Convert to field representation
    print("\n2. Converting to mycelium field...")
    field = returns_to_field(returns, grid_size=64)
    print(f"   Field shape: {field.shape}")
    print(f"   Field range: [{field.min() * 1000:.2f}, {field.max() * 1000:.2f}] mV")

    # Step 3: Detect regime
    print("\n3. Analyzing regime with fractal dynamics...")
    regime_info = detect_regime(field)
    print(f"   Fractal dimension: {regime_info['fractal_dimension']:.4f}")
    print(f"   Volatility: {regime_info['volatility_mV']:.4f} mV")
    print(f"   Detected regime: {regime_info['regime']}")

    # Step 4: Run mycelium simulation
    print("\n4. Running mycelium field simulation...")
    rng_sim = np.random.default_rng(seed)
    evolved_field, growth_events = simulate_mycelium_field(
        rng_sim,
        grid_size=64,
        steps=64,
        turing_enabled=True,
    )
    print(f"   Growth events: {growth_events}")

    # Analyze evolved field
    evolved_regime = detect_regime(evolved_field)
    print(f"   Evolved fractal dim: {evolved_regime['fractal_dimension']:.4f}")

    # Step 5: Generate fractal for stability analysis
    print("\n5. Lyapunov stability analysis...")
    _, lyapunov = generate_fractal_ifs(rng_sim, num_points=5000)
    print(f"   Lyapunov exponent: {lyapunov:.4f}")
    if lyapunov < 0:
        print("   → System is STABLE (contractive dynamics)")
    else:
        print("   → System is UNSTABLE (expansive dynamics)")

    # Step 6: Train model on regime features
    print("\n6. Training MyceliumFractalNet on regime features...")
    model = MyceliumFractalNet(input_dim=4, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # Create simple training data from field statistics
    features = torch.tensor(
        [
            [
                regime_info["fractal_dimension"],
                field.mean() * 100,
                field.std() * 100,
                field.max() * 100,
            ]
        ],
        dtype=torch.float32,
    )
    target = torch.tensor([[1.0 if regime_info["regime"] == "normal" else 0.0]])

    # Train for a few steps
    for epoch in range(10):
        loss = model.train_step(features, target, optimizer, loss_fn)

    print(f"   Final loss: {loss:.6f}")

    # Step 7: Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Market Regime Analysis")
    print("=" * 60)
    print(f"Fractal Dimension: {regime_info['fractal_dimension']:.4f}")
    print(f"Lyapunov Exponent: {lyapunov:.4f} (stable)" if lyapunov < 0 else "unstable")
    print(f"Regime: {regime_info['regime'].upper()}")
    print(f"Growth Events: {growth_events}")
    print("\nInterpretation:")
    if regime_info["regime"] == "high_complexity":
        print("→ Market shows high complexity, expect increased volatility")
    elif regime_info["regime"] == "low_complexity":
        print("→ Market shows low complexity, possible trend/consolidation")
    else:
        print("→ Market in normal regime, standard risk parameters apply")


if __name__ == "__main__":
    main()
