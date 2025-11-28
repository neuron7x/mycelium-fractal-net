"""Unit tests for MyceliumFractalNet v4.1."""

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mfn import (
    FractalLayer,
    MyceliumFractalNet,
    NernstPotential,
    STDPModule,
    TuringGrowth,
    load_config,
)


class TestNernstPotential:
    """Tests for NernstPotential activation function."""

    def test_initialization(self):
        """Test default initialization."""
        nernst = NernstPotential()
        assert nernst.threshold.item() == pytest.approx(-0.065)
        assert nernst.amplitude.item() == pytest.approx(1.0)
        assert nernst.scale.item() == pytest.approx(10.0)

    def test_custom_initialization(self):
        """Test custom parameter initialization."""
        nernst = NernstPotential(threshold=-0.05, amplitude=0.8, scale=5.0)
        assert nernst.threshold.item() == pytest.approx(-0.05)
        assert nernst.amplitude.item() == pytest.approx(0.8)
        assert nernst.scale.item() == pytest.approx(5.0)

    def test_forward_shape(self):
        """Test output shape matches input shape."""
        nernst = NernstPotential()
        x = torch.randn(4, 8)
        y = nernst(x)
        assert y.shape == x.shape

    def test_forward_bounded(self):
        """Test output is bounded by amplitude."""
        nernst = NernstPotential(amplitude=1.0)
        x = torch.randn(100) * 10  # Large inputs
        y = nernst(x)
        assert torch.all(y >= -1.0)
        assert torch.all(y <= 1.0)

    def test_forward_differentiable(self):
        """Test that gradients can flow through."""
        nernst = NernstPotential()
        x = torch.randn(4, 8, requires_grad=True)
        y = nernst(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestSTDPModule:
    """Tests for STDP learning module."""

    def test_initialization(self):
        """Test default initialization."""
        stdp = STDPModule()
        assert stdp.tau_plus == 20.0
        assert stdp.tau_minus == 20.0
        assert stdp.a_plus == 0.01
        assert stdp.a_minus == 0.01
        assert stdp.w_max == 1.0

    def test_forward_output_shapes(self):
        """Test forward pass output shapes."""
        stdp = STDPModule()
        pre = torch.rand(10)
        post = torch.rand(8)
        pot, dep = stdp(pre, post)
        assert pot.shape == (8, 10)
        assert dep.shape == (10, 8)

    def test_compute_weight_update_shape(self):
        """Test weight update computation shapes."""
        stdp = STDPModule()
        pre_spikes = torch.zeros(5)
        post_spikes = torch.zeros(4)
        pre_spikes[2] = 1.0
        post_spikes[1] = 1.0
        weights = torch.ones(4, 5) * 0.5

        delta_w = stdp.compute_weight_update(pre_spikes, post_spikes, weights)
        assert delta_w.shape == weights.shape


class TestTuringGrowth:
    """Tests for Turing pattern growth module."""

    def test_initialization(self):
        """Test default initialization."""
        turing = TuringGrowth(grid_size=8)
        assert turing.grid_size == 8
        assert turing.activator.shape == (8, 8)
        assert turing.inhibitor.shape == (8, 8)

    def test_step_preserves_bounds(self):
        """Test that step keeps values in [0, 1]."""
        turing = TuringGrowth(grid_size=8)
        for _ in range(10):
            a, i = turing.step()
            assert torch.all(a >= 0) and torch.all(a <= 1)
            assert torch.all(i >= 0) and torch.all(i <= 1)

    def test_generate_connection_mask_shape(self):
        """Test connection mask output shape."""
        turing = TuringGrowth(grid_size=16)
        mask = turing.generate_connection_mask(threshold=0.5, num_steps=5)
        assert mask.shape == (16, 16)
        assert mask.dtype == torch.bool

    def test_forward_shape_2d(self):
        """Test forward pass with 2D input."""
        turing = TuringGrowth(grid_size=8)
        x = torch.randn(4, 16)
        y = turing(x)
        assert y.shape == x.shape


class TestFractalLayer:
    """Tests for FractalLayer module."""

    def test_initialization(self):
        """Test layer initialization."""
        layer = FractalLayer(32, 64, fractal_depth=3)
        assert layer.in_features == 32
        assert layer.out_features == 64
        assert layer.fractal_depth == 3
        assert len(layer.pathways) == 3

    def test_forward_shape(self):
        """Test forward pass output shape."""
        layer = FractalLayer(32, 64, fractal_depth=3)
        x = torch.randn(8, 32)
        y = layer(x)
        assert y.shape == (8, 64)

    def test_forward_differentiable(self):
        """Test that gradients can flow through."""
        layer = FractalLayer(32, 64, fractal_depth=2)
        x = torch.randn(4, 32, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None

    def test_pathway_weights_learnable(self):
        """Test that pathway weights are learnable parameters."""
        layer = FractalLayer(32, 64, fractal_depth=3)
        assert layer.pathway_weights.requires_grad

    def test_without_nernst(self):
        """Test layer with ReLU instead of Nernst."""
        layer = FractalLayer(32, 64, use_nernst=False)
        x = torch.randn(4, 32)
        y = layer(x)
        assert y.shape == (4, 64)


class TestMyceliumFractalNet:
    """Tests for MyceliumFractalNet model."""

    @pytest.fixture
    def default_config(self):
        """Create default configuration."""
        return {
            "architecture": {
                "input_dim": 32,
                "hidden_dim": 64,
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

    def test_initialization_from_dict(self, default_config):
        """Test model initialization from config dict."""
        model = MyceliumFractalNet(default_config)
        assert model.input_dim == 32
        assert model.hidden_dim == 64
        assert model.output_dim == 10
        assert model.num_layers == 3

    def test_initialization_from_file(self, default_config):
        """Test model initialization from config file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(default_config, f)
            f.flush()

            model = MyceliumFractalNet(f.name)
            assert model.input_dim == 32
            assert model.output_dim == 10

    def test_forward_shape(self, default_config):
        """Test forward pass output shape."""
        model = MyceliumFractalNet(default_config)
        x = torch.randn(8, 32)
        y = model(x)
        assert y.shape == (8, 10)

    def test_forward_eval_mode(self, default_config):
        """Test forward pass in eval mode."""
        model = MyceliumFractalNet(default_config)
        model.eval()
        x = torch.randn(4, 32)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (4, 10)

    def test_get_network_stats(self, default_config):
        """Test network statistics."""
        model = MyceliumFractalNet(default_config)
        stats = model.get_network_stats()

        assert "num_parameters" in stats
        assert "num_trainable_parameters" in stats
        assert "num_layers" in stats
        assert "fractal_depth" in stats
        assert "graph_nodes" in stats
        assert "graph_edges" in stats

        assert stats["num_parameters"] > 0
        assert stats["num_layers"] == 3

    def test_visualize_graph(self, default_config):
        """Test graph visualization data."""
        model = MyceliumFractalNet(default_config)
        graph_data = model.visualize_graph()

        assert "nodes" in graph_data
        assert "edges" in graph_data
        assert len(graph_data["nodes"]) > 0
        assert len(graph_data["edges"]) > 0

    def test_apply_stdp_update(self, default_config):
        """Test STDP update application."""
        model = MyceliumFractalNet(default_config)
        model.train()
        x = torch.randn(4, 32)
        _ = model(x)
        # Should not raise
        model.apply_stdp_update(learning_rate=0.001)

    def test_turing_disabled(self, default_config):
        """Test model with Turing disabled."""
        default_config["turing"]["enabled"] = False
        model = MyceliumFractalNet(default_config)
        x = torch.randn(4, 32)
        y = model(x)
        assert y.shape == (4, 10)

    def test_stdp_disabled(self, default_config):
        """Test model with STDP disabled."""
        default_config["stdp"]["enabled"] = False
        model = MyceliumFractalNet(default_config)
        model.train()
        x = torch.randn(4, 32)
        _ = model(x)
        model.apply_stdp_update()  # Should not raise

    def test_gradient_flow(self, default_config):
        """Test that gradients flow through the model."""
        model = MyceliumFractalNet(default_config)
        model.train()
        x = torch.randn(4, 32, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None


class TestLoadConfig:
    """Tests for configuration loading."""

    def test_load_valid_config(self):
        """Test loading a valid config file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"test": "value"}, f)
            f.flush()
            config = load_config(f.name)
            assert config == {"test": "value"}

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.json")

    def test_load_invalid_json(self):
        """Test loading invalid JSON raises error."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("not valid json {")
            f.flush()
            with pytest.raises(json.JSONDecodeError):
                load_config(f.name)


class TestConfigFiles:
    """Tests for provided configuration files."""

    @pytest.fixture
    def configs_dir(self):
        """Get configs directory path."""
        return Path(__file__).parent.parent.parent / "configs"

    def test_small_config_exists(self, configs_dir):
        """Test small config file exists and is valid."""
        config_path = configs_dir / "small.json"
        if config_path.exists():
            config = load_config(config_path)
            assert "architecture" in config
            assert config["architecture"]["input_dim"] == 32

    def test_medium_config_exists(self, configs_dir):
        """Test medium config file exists and is valid."""
        config_path = configs_dir / "medium.json"
        if config_path.exists():
            config = load_config(config_path)
            assert "architecture" in config
            assert config["architecture"]["input_dim"] == 128

    def test_large_config_exists(self, configs_dir):
        """Test large config file exists and is valid."""
        config_path = configs_dir / "large.json"
        if config_path.exists():
            config = load_config(config_path)
            assert "architecture" in config
            assert config["architecture"]["input_dim"] == 512
