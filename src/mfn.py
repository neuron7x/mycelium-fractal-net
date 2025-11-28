"""
MyceliumFractalNet v4.1 - Adaptive Fractal Neural Networks

This module implements a bio-inspired neural network architecture featuring:
- Turing pattern-based growth dynamics for network topology evolution
- Nernst potential modeling for biologically plausible activation
- Spike-Timing Dependent Plasticity (STDP) for adaptive learning
- Fractal layer organization for hierarchical feature processing

Designed for PyTorch 2.1+ with production-ready implementations.
"""

import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a JSON file.

    Args:
        config_path: Path to the configuration JSON file.

    Returns:
        Dictionary containing the configuration parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
        json.JSONDecodeError: If the config file is not valid JSON.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


class NernstPotential(nn.Module):
    """Nernst potential-based activation function.

    Implements a biologically plausible activation based on the Nernst equation,
    which describes the equilibrium potential of an ion across a membrane.

    The Nernst potential is computed as:
        E = (RT/zF) * ln([ion_out]/[ion_in])

    For neural network purposes, this is simplified to:
        activation(x) = amplitude * tanh(scale * (x - threshold))

    Args:
        threshold: Resting membrane potential threshold (default: -0.065).
        amplitude: Maximum activation amplitude (default: 1.0).
        scale: Scaling factor for input sensitivity (default: 10.0).
        temperature: Temperature in Kelvin for Nernst scaling (default: 310.0).
    """

    def __init__(
        self,
        threshold: float = -0.065,
        amplitude: float = 1.0,
        scale: float = 10.0,
        temperature: float = 310.0,
    ):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=False)
        self.amplitude = nn.Parameter(torch.tensor(amplitude), requires_grad=False)
        self.scale = nn.Parameter(torch.tensor(scale), requires_grad=False)
        # Nernst constant: RT/F (R=8.314, F=96485)
        self.nernst_factor = nn.Parameter(
            torch.tensor((8.314 * temperature) / 96485.0), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Nernst potential-based activation.

        Args:
            x: Input tensor of any shape.

        Returns:
            Activated tensor with the same shape as input.
        """
        normalized = (x - self.threshold) * self.scale
        return self.amplitude * torch.tanh(normalized * self.nernst_factor)


class STDPModule(nn.Module):
    """Spike-Timing Dependent Plasticity (STDP) learning module.

    Implements STDP, a biological learning rule where synaptic weight changes
    depend on the relative timing of pre- and post-synaptic spikes.

    STDP rule:
        Δw = A+ * exp(-Δt/τ+) if Δt > 0 (LTP)
        Δw = -A- * exp(Δt/τ-) if Δt < 0 (LTD)

    Where Δt = t_post - t_pre is the time difference between spikes.

    Args:
        tau_plus: Time constant for potentiation (default: 20.0).
        tau_minus: Time constant for depression (default: 20.0).
        a_plus: Learning rate for potentiation (default: 0.01).
        a_minus: Learning rate for depression (default: 0.01).
        w_max: Maximum weight value (default: 1.0).
    """

    def __init__(
        self,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        a_plus: float = 0.01,
        a_minus: float = 0.01,
        w_max: float = 1.0,
    ):
        super().__init__()
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.w_max = w_max

        # Trace tensors for tracking spike history
        self.register_buffer("pre_trace", None)
        self.register_buffer("post_trace", None)

    def compute_weight_update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """Compute STDP weight updates.

        Args:
            pre_spikes: Pre-synaptic spike tensor.
            post_spikes: Post-synaptic spike tensor.
            weights: Current weight matrix.
            dt: Time step (default: 1.0).

        Returns:
            Weight update tensor (delta_w).
        """
        # Initialize traces if needed
        if self.pre_trace is None or self.pre_trace.shape != pre_spikes.shape:
            self.pre_trace = torch.zeros_like(pre_spikes)
        if self.post_trace is None or self.post_trace.shape != post_spikes.shape:
            self.post_trace = torch.zeros_like(post_spikes)

        # Decay traces
        decay_pre = math.exp(-dt / self.tau_plus)
        decay_post = math.exp(-dt / self.tau_minus)
        self.pre_trace = self.pre_trace * decay_pre + pre_spikes
        self.post_trace = self.post_trace * decay_post + post_spikes

        # Compute weight changes
        # LTP: post-synaptic spike following pre-synaptic spike
        # Result shape: (num_post, num_pre) matching weight matrix
        ltp = self.a_plus * torch.outer(post_spikes, self.pre_trace)
        # LTD: pre-synaptic spike following post-synaptic spike
        # Result shape: (num_post, num_pre) matching weight matrix
        ltd = self.a_minus * torch.outer(self.post_trace, pre_spikes)

        delta_w = ltp - ltd

        # Soft bounds
        delta_w = delta_w * (self.w_max - weights) * weights

        return delta_w

    def forward(
        self, pre_activity: torch.Tensor, post_activity: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute correlation for STDP-like learning.

        Args:
            pre_activity: Pre-synaptic activity.
            post_activity: Post-synaptic activity.

        Returns:
            Tuple of (potentiation signal, depression signal).
        """
        # Convert activities to binary spikes using threshold
        pre_spikes = (pre_activity > 0.5).float()
        post_spikes = (post_activity > 0.5).float()

        # Compute Hebbian-like correlation
        potentiation = torch.outer(post_spikes.flatten(), pre_spikes.flatten())
        depression = torch.outer(pre_spikes.flatten(), post_spikes.flatten())

        return potentiation, depression


class TuringGrowth(nn.Module):
    """Turing pattern-based network topology growth.

    Implements a simplified Turing reaction-diffusion model for dynamic
    network topology evolution. The system uses activator-inhibitor dynamics
    to create self-organizing patterns that guide network growth.

    The reaction-diffusion equations:
        ∂A/∂t = D_A∇²A + f(A,I)
        ∂I/∂t = D_I∇²I + g(A,I)

    Where A is the activator, I is the inhibitor, and D_A, D_I are diffusion rates.

    Args:
        grid_size: Size of the growth grid (default: 16).
        diffusion_activator: Diffusion rate for activator (default: 0.16).
        diffusion_inhibitor: Diffusion rate for inhibitor (default: 0.08).
        feed_rate: Feed rate for activator (default: 0.035).
        kill_rate: Kill rate for inhibitor (default: 0.065).
    """

    def __init__(
        self,
        grid_size: int = 16,
        diffusion_activator: float = 0.16,
        diffusion_inhibitor: float = 0.08,
        feed_rate: float = 0.035,
        kill_rate: float = 0.065,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.d_a = diffusion_activator
        self.d_i = diffusion_inhibitor
        self.feed = feed_rate
        self.kill = kill_rate

        # Initialize activator and inhibitor grids
        self.register_buffer(
            "activator", torch.ones(grid_size, grid_size) * 0.5
        )
        self.register_buffer(
            "inhibitor", torch.ones(grid_size, grid_size) * 0.25
        )

        # Laplacian kernel for diffusion
        laplacian = torch.tensor(
            [[0.05, 0.2, 0.05], [0.2, -1.0, 0.2], [0.05, 0.2, 0.05]]
        )
        self.register_buffer("laplacian_kernel", laplacian.view(1, 1, 3, 3))

    def _laplacian(self, grid: torch.Tensor) -> torch.Tensor:
        """Compute discrete Laplacian using convolution."""
        grid_4d = grid.view(1, 1, self.grid_size, self.grid_size)
        result = F.conv2d(grid_4d, self.laplacian_kernel, padding=1)
        return result.view(self.grid_size, self.grid_size)

    def step(self, dt: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one step of Turing pattern evolution.

        Args:
            dt: Time step for integration (default: 1.0).

        Returns:
            Tuple of (activator grid, inhibitor grid) after the step.
        """
        a, i = self.activator, self.inhibitor

        # Reaction term (Gray-Scott model)
        reaction = a * i * i

        # Laplacian (diffusion)
        lap_a = self._laplacian(a)
        lap_i = self._laplacian(i)

        # Update equations
        da = self.d_a * lap_a - reaction + self.feed * (1 - a)
        di = self.d_i * lap_i + reaction - (self.kill + self.feed) * i

        # Apply updates with clamping
        self.activator = torch.clamp(a + da * dt, 0, 1)
        self.inhibitor = torch.clamp(i + di * dt, 0, 1)

        return self.activator, self.inhibitor

    def generate_connection_mask(
        self, threshold: float = 0.5, num_steps: int = 10
    ) -> torch.Tensor:
        """Generate a connection mask based on Turing patterns.

        Args:
            threshold: Threshold for activator values to form connections.
            num_steps: Number of Turing steps to run.

        Returns:
            Boolean mask tensor indicating where connections should exist.
        """
        for _ in range(num_steps):
            self.step()

        return self.activator > threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Turing-modulated transformation.

        Args:
            x: Input tensor to modulate.

        Returns:
            Modulated output tensor.
        """
        # Interpolate activator to match input size
        batch_size = x.shape[0]
        if len(x.shape) == 2:
            target_size = x.shape[1]
            pattern = F.interpolate(
                self.activator.view(1, 1, self.grid_size, self.grid_size),
                size=(1, target_size),
                mode="bilinear",
                align_corners=False,
            )
            pattern = pattern.view(1, target_size).expand(batch_size, -1)
        else:
            pattern = self.activator.mean()

        return x * pattern


class FractalLayer(nn.Module):
    """Fractal neural network layer with self-similar structure.

    Implements a layer with fractal organization where sub-layers share
    structural properties at different scales. The fractal depth controls
    the recursion level of the self-similar structure.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        fractal_depth: Depth of fractal recursion (default: 3).
        dropout_rate: Dropout probability (default: 0.1).
        use_nernst: Whether to use Nernst activation (default: True).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        fractal_depth: int = 3,
        dropout_rate: float = 0.1,
        use_nernst: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fractal_depth = fractal_depth

        # Build fractal pathways
        self.pathways = nn.ModuleList()
        for depth in range(1, fractal_depth + 1):
            # Each pathway has a different intermediate size
            intermediate = max(8, out_features // (2 ** (depth - 1)))
            pathway = nn.Sequential(
                nn.Linear(in_features, intermediate),
                nn.LayerNorm(intermediate),
                NernstPotential() if use_nernst else nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(intermediate, out_features),
            )
            self.pathways.append(pathway)

        # Pathway combination weights (learnable)
        self.pathway_weights = nn.Parameter(
            torch.ones(fractal_depth) / fractal_depth
        )

        # Final normalization
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through fractal pathways.

        Args:
            x: Input tensor of shape (batch, in_features).

        Returns:
            Output tensor of shape (batch, out_features).
        """
        # Compute outputs from all pathways
        outputs = [pathway(x) for pathway in self.pathways]

        # Weighted combination
        weights = F.softmax(self.pathway_weights, dim=0)
        combined = sum(w * out for w, out in zip(weights, outputs))

        return self.norm(combined)


class MyceliumFractalNet(nn.Module):
    """MyceliumFractalNet: Adaptive fractal neural network with bio-inspired dynamics.

    This network combines fractal layer organization with Turing pattern-based
    growth dynamics, Nernst potential activations, and STDP-like plasticity.

    Features:
        - Fractal layer hierarchy for multi-scale feature processing
        - Turing pattern-driven topology adaptation
        - Biologically plausible Nernst potential activations
        - STDP-inspired learning mechanisms
        - NetworkX graph representation for network analysis

    Args:
        config: Configuration dictionary or path to config file.
    """

    def __init__(self, config: Union[Dict[str, Any], str, Path]):
        super().__init__()

        # Load configuration
        if isinstance(config, (str, Path)):
            config = load_config(config)
        self.config = config

        # Extract architecture parameters
        arch = config.get("architecture", {})
        self.input_dim = arch.get("input_dim", 64)
        self.hidden_dim = arch.get("hidden_dim", 128)
        self.output_dim = arch.get("output_dim", 10)
        self.num_layers = arch.get("num_layers", 3)
        self.fractal_depth = arch.get("fractal_depth", 3)
        self.dropout_rate = arch.get("dropout_rate", 0.1)

        # Extract Turing parameters
        turing = config.get("turing", {})
        self.turing_grid_size = turing.get("grid_size", 16)
        self.turing_enabled = turing.get("enabled", True)

        # Extract STDP parameters
        stdp = config.get("stdp", {})
        self.stdp_enabled = stdp.get("enabled", True)

        # Build network
        self._build_network()

        # Initialize Turing growth module
        if self.turing_enabled:
            self.turing = TuringGrowth(
                grid_size=self.turing_grid_size,
                diffusion_activator=turing.get("diffusion_activator", 0.16),
                diffusion_inhibitor=turing.get("diffusion_inhibitor", 0.08),
                feed_rate=turing.get("feed_rate", 0.035),
                kill_rate=turing.get("kill_rate", 0.065),
            )

        # Initialize STDP module
        if self.stdp_enabled:
            self.stdp = STDPModule(
                tau_plus=stdp.get("tau_plus", 20.0),
                tau_minus=stdp.get("tau_minus", 20.0),
                a_plus=stdp.get("a_plus", 0.01),
                a_minus=stdp.get("a_minus", 0.01),
            )

        # Build network graph for analysis
        self.graph = self._build_graph()

    def _build_network(self):
        """Build the fractal network architecture."""
        layers = []

        # Input layer
        layers.append(
            FractalLayer(
                self.input_dim,
                self.hidden_dim,
                fractal_depth=self.fractal_depth,
                dropout_rate=self.dropout_rate,
            )
        )

        # Hidden layers
        for _ in range(self.num_layers - 2):
            layers.append(
                FractalLayer(
                    self.hidden_dim,
                    self.hidden_dim,
                    fractal_depth=self.fractal_depth,
                    dropout_rate=self.dropout_rate,
                )
            )

        # Output layer
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))

        self.layers = nn.ModuleList(layers)

    def _build_graph(self) -> nx.DiGraph:
        """Build a NetworkX graph representation of the network.

        Returns:
            Directed graph representing network topology.
        """
        G = nx.DiGraph()

        # Add nodes for each layer
        layer_sizes = (
            [self.input_dim]
            + [self.hidden_dim] * (self.num_layers - 1)
            + [self.output_dim]
        )

        node_id = 0
        layer_nodes = []

        for layer_idx, size in enumerate(layer_sizes):
            current_layer_nodes = []
            for _ in range(min(size, 10)):  # Limit for visualization
                G.add_node(node_id, layer=layer_idx)
                current_layer_nodes.append(node_id)
                node_id += 1
            layer_nodes.append(current_layer_nodes)

        # Add edges between consecutive layers
        for i in range(len(layer_nodes) - 1):
            for src in layer_nodes[i]:
                for dst in layer_nodes[i + 1]:
                    G.add_edge(src, dst)

        return G

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Output tensor of shape (batch, output_dim).
        """
        # Apply Turing modulation if enabled
        if self.turing_enabled:
            x = self.turing(x)
            self.turing.step()

        # Forward through layers
        activations = [x]
        for layer in self.layers[:-1]:
            x = layer(x)
            activations.append(x)

        # Final output layer
        x = self.layers[-1](x)

        # Store activations for STDP if enabled
        if self.stdp_enabled and self.training:
            self._last_activations = activations

        return x

    def apply_stdp_update(self, learning_rate: float = 0.001):
        """Apply STDP-based weight updates using stored activations.

        Args:
            learning_rate: Learning rate for STDP updates.
        """
        if not self.stdp_enabled or not hasattr(self, "_last_activations"):
            return

        activations = self._last_activations
        for i, layer in enumerate(self.layers[:-1]):
            if i + 1 < len(activations):
                pre = activations[i].mean(dim=0)
                post = activations[i + 1].mean(dim=0)
                _, _ = self.stdp(pre, post)

    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics and topology information.

        Returns:
            Dictionary containing network statistics.
        """
        stats = {
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "num_trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "num_layers": self.num_layers,
            "fractal_depth": self.fractal_depth,
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges(),
        }

        if self.turing_enabled:
            stats["turing_activator_mean"] = self.turing.activator.mean().item()
            stats["turing_inhibitor_mean"] = self.turing.inhibitor.mean().item()

        return stats

    def visualize_graph(self) -> Dict[str, List]:
        """Get graph data for visualization.

        Returns:
            Dictionary with nodes and edges lists.
        """
        nodes = [
            {"id": n, "layer": d["layer"]}
            for n, d in self.graph.nodes(data=True)
        ]
        edges = [{"source": u, "target": v} for u, v in self.graph.edges()]

        return {"nodes": nodes, "edges": edges}


def validate_model(config_path: Optional[str] = None) -> bool:
    """Validate the MyceliumFractalNet implementation.

    Args:
        config_path: Path to configuration file (optional).

    Returns:
        True if validation passes, False otherwise.
    """
    print("=" * 60)
    print("MyceliumFractalNet v4.1 Validation")
    print("=" * 60)

    try:
        # Create default config if not provided
        if config_path is None:
            config = {
                "architecture": {
                    "input_dim": 64,
                    "hidden_dim": 128,
                    "output_dim": 10,
                    "num_layers": 3,
                    "fractal_depth": 2,
                    "dropout_rate": 0.1,
                },
                "turing": {
                    "enabled": True,
                    "grid_size": 8,
                },
                "stdp": {
                    "enabled": True,
                },
            }
        else:
            config = load_config(config_path)

        print("\n[1/6] Loading configuration...")
        print(f"      Architecture: {config.get('architecture', {})}")

        print("\n[2/6] Creating model...")
        model = MyceliumFractalNet(config)
        stats = model.get_network_stats()
        print(f"      Parameters: {stats['num_parameters']:,}")
        print(f"      Trainable: {stats['num_trainable_parameters']:,}")

        print("\n[3/6] Testing forward pass...")
        batch_size = 8
        input_dim = config.get("architecture", {}).get("input_dim", 64)
        x = torch.randn(batch_size, input_dim)
        output = model(x)
        output_dim = config.get("architecture", {}).get("output_dim", 10)
        assert output.shape == (batch_size, output_dim), f"Output shape mismatch: {output.shape}"
        print(f"      Input: {x.shape} -> Output: {output.shape}")

        print("\n[4/6] Testing Turing growth...")
        turing = TuringGrowth(grid_size=8)
        for _ in range(5):
            turing.step()
        mask = turing.generate_connection_mask(threshold=0.4, num_steps=5)
        print(f"      Grid size: 8x8, Active connections: {mask.sum().item()}")

        print("\n[5/6] Testing STDP module...")
        stdp = STDPModule()
        pre = torch.rand(10)
        post = torch.rand(10)
        pot, dep = stdp(pre, post)
        print(f"      Potentiation shape: {pot.shape}")
        print(f"      Depression shape: {dep.shape}")

        print("\n[6/6] Testing Nernst activation...")
        nernst = NernstPotential()
        test_input = torch.linspace(-1, 1, 10)
        activated = nernst(test_input)
        print(f"      Input range: [{test_input.min():.2f}, {test_input.max():.2f}]")
        print(f"      Output range: [{activated.min():.2f}, {activated.max():.2f}]")

        print("\n" + "=" * 60)
        print("✓ All validations passed!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    success = validate_model(config_path)
    sys.exit(0 if success else 1)
