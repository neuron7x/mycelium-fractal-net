# MyceliumFractalNet v4.1

[![CI](https://github.com/neuron7x/mycelium-fractal-net/actions/workflows/ci.yml/badge.svg)](https://github.com/neuron7x/mycelium-fractal-net/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Adaptive fractal neural networks with bio-inspired dynamics** – Turing growth, Nernst potentials, STDP plasticity. Production-ready for finance, molecular, RL, and medical applications.

## 🚀 Quick Start

**Validate in 30 seconds:**

```bash
# Clone and install
git clone https://github.com/neuron7x/mycelium-fractal-net.git
cd mycelium-fractal-net
pip install -r requirements.txt

# Run 1-click validation
python validate.py
```

## 📦 Installation

### Requirements

- Python 3.9+
- PyTorch 2.1+
- NumPy
- NetworkX

### Install from source

```bash
git clone https://github.com/neuron7x/mycelium-fractal-net.git
cd mycelium-fractal-net
pip install -e .
```

### Install dependencies only

```bash
pip install -r requirements.txt
```

## 🧬 Features

### Bio-Inspired Dynamics

- **Turing Pattern Growth**: Self-organizing network topology evolution using reaction-diffusion dynamics
- **Nernst Potential Activation**: Biologically plausible membrane potential-based activation functions
- **STDP Plasticity**: Spike-Timing Dependent Plasticity for adaptive synaptic learning

### Fractal Architecture

- **Self-Similar Layers**: Multi-scale feature processing with fractal depth control
- **Adaptive Pathways**: Learnable pathway combination weights
- **NetworkX Integration**: Graph-based network analysis and visualization

## 💡 Usage

### Basic Example

```python
from src.mfn import MyceliumFractalNet
import torch

# Define configuration
config = {
    "architecture": {
        "input_dim": 64,
        "hidden_dim": 128,
        "output_dim": 10,
        "num_layers": 3,
        "fractal_depth": 2,
    },
    "turing": {"enabled": True, "grid_size": 8},
    "stdp": {"enabled": True},
}

# Create model
model = MyceliumFractalNet(config)

# Forward pass
x = torch.randn(32, 64)  # batch_size=32, input_dim=64
output = model(x)
print(f"Output shape: {output.shape}")  # [32, 10]

# Get network statistics
stats = model.get_network_stats()
print(f"Parameters: {stats['num_parameters']:,}")
```

### Using Configuration Files

```python
from src.mfn import MyceliumFractalNet

# Load from config file
model = MyceliumFractalNet("configs/medium.json")

# Or use predefined sizes:
# - configs/small.json  (quick experiments)
# - configs/medium.json (standard training)
# - configs/large.json  (high-capacity models)
```

### Finance Example

```bash
python examples/finance.py
```

This demonstrates time series prediction with synthetic financial data.

## 📁 Project Structure

```
mycelium-fractal-net/
├── src/
│   ├── __init__.py
│   └── mfn.py              # Core implementation
├── tests/
│   └── unit/
│       └── test_mfn.py     # Unit tests
├── configs/
│   ├── small.json          # Small model config
│   ├── medium.json         # Medium model config
│   └── large.json          # Large model config
├── examples/
│   └── finance.py          # Finance prediction example
├── docs/
│   └── math_summary.md     # Mathematical foundations
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI
├── requirements.txt
├── setup.py
├── validate.py             # 1-click validation
├── LICENSE
└── README.md
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src

# Run validation script
python validate.py
```

## 📊 Model Configurations

| Config | Input | Hidden | Output | Layers | Params |
|--------|-------|--------|--------|--------|--------|
| Small  | 32    | 64     | 10     | 2      | ~15K   |
| Medium | 128   | 256    | 64     | 4      | ~500K  |
| Large  | 512   | 1024   | 256    | 6      | ~8M    |

## 📖 Documentation

- [Mathematical Foundations](docs/math_summary.md) - Detailed math behind Turing growth, Nernst potentials, and STDP

## 🔬 Core Components

### NernstPotential

Biologically plausible activation based on membrane potential dynamics:

```python
from src.mfn import NernstPotential

activation = NernstPotential(threshold=-0.065, amplitude=1.0)
output = activation(input_tensor)
```

### TuringGrowth

Self-organizing topology using reaction-diffusion patterns:

```python
from src.mfn import TuringGrowth

turing = TuringGrowth(grid_size=16)
turing.step()  # Evolve patterns
mask = turing.generate_connection_mask(threshold=0.5)
```

### STDPModule

Spike-timing dependent plasticity for adaptive learning:

```python
from src.mfn import STDPModule

stdp = STDPModule(tau_plus=20.0, tau_minus=20.0)
potentiation, depression = stdp(pre_activity, post_activity)
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 Citation

If you use MyceliumFractalNet in your research, please cite:

```bibtex
@software{myceliumfractalnet2025,
  title = {MyceliumFractalNet: Adaptive Fractal Neural Networks with Bio-Inspired Dynamics},
  version = {4.1.0},
  year = {2025},
  url = {https://github.com/neuron7x/mycelium-fractal-net}
}
```
