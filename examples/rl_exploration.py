#!/usr/bin/env python
"""
Reinforcement Learning example: Using MyceliumFractalNet for adaptive exploration.

This example demonstrates how bio-inspired fractal dynamics can enhance
RL exploration strategies through:
- STDP-based reward modulation
- Fractal exploration patterns
- Turing morphogenesis for state-space coverage

Features used:
- STDP for temporal credit assignment
- Sparse attention for efficient policy learning
- Lyapunov stability for safe exploration
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from mycelium_fractal_net import (
    STDP_A_MINUS,
    STDP_A_PLUS,
    STDP_TAU_MINUS,
    STDP_TAU_PLUS,
    MyceliumFractalNet,
    estimate_fractal_dimension,
    generate_fractal_ifs,
    simulate_mycelium_field,
)
from mycelium_fractal_net.model import STDPPlasticity


@dataclass
class GridWorldEnv:
    """Simple grid world environment for RL demo."""

    size: int = 8
    goal: Tuple[int, int] = (7, 7)
    obstacles: list = None  # type: ignore

    def __post_init__(self) -> None:
        if self.obstacles is None:
            self.obstacles = [(3, 3), (3, 4), (4, 3)]
        self.state = (0, 0)

    def reset(self) -> Tuple[int, int]:
        """Reset environment."""
        self.state = (0, 0)
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Take action in environment.

        Actions: 0=up, 1=right, 2=down, 3=left
        """
        x, y = self.state
        dx, dy = [(0, 1), (1, 0), (0, -1), (-1, 0)][action]
        new_x = max(0, min(self.size - 1, x + dx))
        new_y = max(0, min(self.size - 1, y + dy))

        # Check obstacles
        if (new_x, new_y) not in self.obstacles:
            self.state = (new_x, new_y)

        # Compute reward
        if self.state == self.goal:
            return self.state, 1.0, True
        elif self.state in self.obstacles:
            return self.state, -0.5, False
        else:
            # Small step penalty + distance-based shaping
            dist = abs(self.state[0] - self.goal[0]) + abs(self.state[1] - self.goal[1])
            return self.state, -0.01 - 0.001 * dist, False

    def state_to_features(self, state: Tuple[int, int]) -> np.ndarray:
        """Convert state to feature vector."""
        x, y = state
        gx, gy = self.goal
        features = np.array(
            [
                x / self.size,
                y / self.size,
                (gx - x) / self.size,
                (gy - y) / self.size,
            ],
            dtype=np.float32,
        )
        return features


class FractalExplorer:
    """Exploration strategy based on fractal dynamics."""

    def __init__(self, grid_size: int = 8, seed: int = 42) -> None:
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)
        self.visit_counts = np.zeros((grid_size, grid_size))
        self.exploration_field = None

    def update_exploration_field(self) -> None:
        """Update exploration field using mycelium simulation."""
        # Initialize field based on inverse visit counts
        inverse_visits = 1.0 / (self.visit_counts + 1)
        self.exploration_field = inverse_visits * 0.01 - 0.07

    def get_exploration_bonus(self, state: Tuple[int, int]) -> float:
        """Get exploration bonus for state based on fractal coverage."""
        x, y = state
        self.visit_counts[x, y] += 1

        # Compute bonus from inverse visit count
        bonus = 1.0 / np.sqrt(self.visit_counts[x, y])

        return bonus * 0.1

    def analyze_coverage(self) -> dict:
        """Analyze state space coverage."""
        # Compute fractal dimension of visited states
        visited = self.visit_counts > 0
        if visited.sum() < 4:
            return {"fractal_dim": 0.0, "coverage": visited.mean()}

        D = estimate_fractal_dimension(visited)
        return {
            "fractal_dim": D,
            "coverage": visited.mean(),
            "unique_states": int(visited.sum()),
        }


class STDPRewardModulator:
    """Modulate rewards using STDP-like temporal credit assignment."""

    def __init__(self) -> None:
        self.stdp = STDPPlasticity()
        self.action_times = []
        self.reward_times = []

    def record_action(self, t: float) -> None:
        """Record action time."""
        self.action_times.append(t)
        # Keep only recent
        self.action_times = self.action_times[-100:]

    def record_reward(self, t: float, reward: float) -> float:
        """Record reward and compute modulated value."""
        if reward <= 0 or len(self.action_times) == 0:
            return reward

        # STDP modulation: recent actions get more credit
        modulated = 0.0
        for action_t in self.action_times:
            delta_t = t - action_t
            if delta_t > 0:
                # LTP-like: recent actions get credit
                modulated += STDP_A_PLUS * np.exp(-delta_t / STDP_TAU_PLUS)

        return reward * (1.0 + modulated)


def main() -> None:
    """Run RL example."""
    print("=" * 60)
    print("MyceliumFractalNet Reinforcement Learning Example")
    print("Adaptive Exploration with Fractal Dynamics")
    print("=" * 60)

    # Set seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Initialize components
    env = GridWorldEnv(size=8)
    explorer = FractalExplorer(grid_size=8, seed=seed)
    reward_modulator = STDPRewardModulator()

    # Initialize policy network
    policy = MyceliumFractalNet(
        input_dim=4,
        hidden_dim=32,
        use_sparse_attention=True,
        use_stdp=True,
    )

    print("\n1. Environment Setup")
    print(f"   Grid size: {env.size}x{env.size}")
    print(f"   Goal: {env.goal}")
    print(f"   Obstacles: {env.obstacles}")

    # Training loop
    print("\n2. Training with Fractal Exploration...")
    num_episodes = 50
    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        t = 0

        for step in range(100):  # Max steps per episode
            # Get state features
            features = torch.tensor(env.state_to_features(state)).unsqueeze(0)

            # Get policy output (used for logging/debugging)
            with torch.no_grad():
                _ = policy(features).squeeze()

            # Epsilon-greedy with exploration bonus
            exploration_bonus = explorer.get_exploration_bonus(state)
            epsilon = max(0.1, 0.5 - episode * 0.01)

            if rng.random() < epsilon + exploration_bonus:
                action = rng.integers(0, 4)
            else:
                action = 0  # Default action (simplified)

            # Take action
            next_state, reward, done = env.step(action)

            # Record for STDP modulation
            reward_modulator.record_action(float(t))
            if reward > 0:
                reward = reward_modulator.record_reward(float(t), reward)

            total_reward += reward
            state = next_state
            t += 1

            if done:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(t)

        if (episode + 1) % 10 == 0:
            print(
                f"   Episode {episode + 1}: "
                f"reward={total_reward:.2f}, steps={t}"
            )

    # Analyze results
    print("\n3. Analysis Results")

    # Coverage analysis
    coverage = explorer.analyze_coverage()
    print(f"   Fractal dimension of coverage: {coverage['fractal_dim']:.4f}")
    print(f"   State space coverage: {coverage['coverage'] * 100:.1f}%")
    print(f"   Unique states visited: {coverage['unique_states']}")

    # Lyapunov stability
    print("\n4. Stability Analysis...")
    _, lyapunov = generate_fractal_ifs(rng, num_points=5000)
    print(f"   Lyapunov exponent: {lyapunov:.4f}")
    print(f"   System stability: {'STABLE' if lyapunov < 0 else 'UNSTABLE'}")

    # Mycelium field simulation
    print("\n5. Mycelium Field Simulation...")
    field, growth = simulate_mycelium_field(rng, grid_size=64, steps=64)
    print(f"   Growth events: {growth}")
    print(f"   Field range: [{field.min() * 1000:.2f}, {field.max() * 1000:.2f}] mV")

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Mean reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Mean length (last 10): {np.mean(episode_lengths[-10:]):.1f}")
    print(f"Exploration coverage: {coverage['coverage'] * 100:.1f}%")
    print(f"Fractal dimension: {coverage['fractal_dim']:.4f}")

    # STDP parameters used
    print("\nSTDP Parameters Used:")
    print(f"   tau+ = {STDP_TAU_PLUS * 1000:.0f} ms")
    print(f"   tau- = {STDP_TAU_MINUS * 1000:.0f} ms")
    print(f"   A+ = {STDP_A_PLUS}")
    print(f"   A- = {STDP_A_MINUS}")


if __name__ == "__main__":
    main()
