<p align="center">
  <img src="assets/header.svg" alt="MyceliumFractalNet" width="100%" />
</p>

<h1 align="center">MyceliumFractalNet v4.1</h1>

<p align="center">
  <strong>Нейрофізична обчислювальна платформа</strong><br>
  Адаптивні мережі • Фрактальна динаміка • Федеративне навчання
</p>

<p align="center">
  <img src="https://img.shields.io/badge/v4.1.0-stable-0969da?style=flat-square" alt="v4.1.0" />
  <img src="https://img.shields.io/badge/Python-≥3.10-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-≥2.0-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/tests-passing-2da44e?style=flat-square" alt="Tests" />
  <img src="https://img.shields.io/badge/license-MIT-97ca00?style=flat-square" alt="MIT" />
</p>

<p align="center">
  <img src="assets/morphogenesis.gif" alt="Turing morphogenesis" width="380" />
</p>

---

## Архітектура

```
┌────────────────────────────────────────────────────────────────────┐
│                     MyceliumFractalNet v4.1                        │
├──────────────────┬──────────────────┬──────────────────────────────┤
│   Nernst-Planck  │      Turing      │      Federated Learning      │
│  Electrochemistry│   Morphogenesis  │       Byzantine-Krum         │
├──────────────────┼──────────────────┼──────────────────────────────┤
│  E = RT/zF·ln()  │  ∂a/∂t = D∇²a+f  │    Krum(g₁...gₙ) → g*        │
│  K⁺: -89 mV      │  threshold: 0.75 │    tolerance: 20%            │
└──────────────────┴──────────────────┴──────────────────────────────┘
```

---

## Валідовані параметри

| Модуль | Параметр | Значення | Одиниці |
|:-------|:---------|:---------|:--------|
| **Nernst** | R | 8.314 | J/(mol·K) |
| | F | 96485.33 | C/mol |
| | T | 310 | K |
| | E_K (K⁺) | −89.01 | mV |
| **Turing** | D_a | 0.1 | grid²/step |
| | D_i | 0.05 | grid²/step |
| | threshold | 0.75 | — |
| **STDP** | τ± | 20 | ms |
| | A+ | 0.01 | — |
| | A− | 0.012 | — |
| **Attention** | top-k | 4 | — |
| **Federated** | clusters | 100 | — |
| | byzantine_f | 0.2 | — |

---

## Модулі

### Nernst-Planck

Мембранний потенціал іона:

$$E = \frac{RT}{zF} \ln\left(\frac{[ion]_{out}}{[ion]_{in}}\right)$$

```python
from mycelium_fractal_net import compute_nernst_potential

E_K = compute_nernst_potential(
    z_valence=1,
    concentration_out_molar=5e-3,   # [K⁺]out = 5 mM
    concentration_in_molar=140e-3,  # [K⁺]in = 140 mM
    temperature_k=310.0             # 37°C
)
# E_K = -0.08901 V ≈ -89 mV
```

<p align="center">
  <img src="assets/node_dynamics.png" alt="Node dynamics" width="550" />
</p>

### Turing Morphogenesis

Реакційно-дифузійна система:

$$\frac{\partial a}{\partial t} = D_a \nabla^2 a + r_a \cdot a(1-a) - i$$

$$\frac{\partial i}{\partial t} = D_i \nabla^2 i + r_i \cdot (a - i)$$

```python
from mycelium_fractal_net import simulate_mycelium_field
import numpy as np

rng = np.random.default_rng(42)
field, growth_events = simulate_mycelium_field(
    rng=rng,
    grid_size=64,
    steps=64,
    turing_enabled=True
)
# field: [-95, 40] mV range
# growth_events: ~20 per simulation
```

### Fractal Analysis

Box-counting розмірність:

$$D = \lim_{\epsilon \to 0} \frac{\ln N(\epsilon)}{\ln(1/\epsilon)}$$

<p align="center">
  <img src="assets/fractal_topology.png" alt="Fractal topology" width="380" />
</p>

```python
from mycelium_fractal_net import estimate_fractal_dimension

binary = field > -0.060  # threshold -60 mV
D = estimate_fractal_dimension(binary)
# D ∈ [1.4, 1.9]
```

---

## Встановлення

```bash
git clone https://github.com/neuron7x/mycelium-fractal-net.git
cd mycelium-fractal-net
pip install -e ".[dev]"
```

## CLI

```bash
python mycelium_fractal_net_v4_1.py --mode validate --seed 42 --epochs 5
```

```
=== MyceliumFractalNet v4.1 :: validation ===
loss_start              :  2.432786
loss_final              :  0.249718
loss_drop               :  2.183068
pot_min_mV              : -71.083952
pot_max_mV              : -62.975776
lyapunov_exponent       : -2.121279
nernst_symbolic_mV      : -89.010669
```

## API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Input | Output |
|:---------|:-------|:------|:-------|
| `/health` | GET | — | `{status, version}` |
| `/validate` | POST | `{seed, epochs, grid_size}` | `{loss_*, pot_*, fractal_dim}` |
| `/simulate` | POST | `{seed, grid_size, steps}` | `{field_stats, growth_events}` |
| `/nernst` | POST | `{z_valence, concentration_out_molar, concentration_in_molar, temperature_k}` | `{potential_mV}` |
| `/federated/aggregate` | POST | `{gradients[], num_clusters, byzantine_fraction}` | `{aggregated_gradient}` |

---

## Docker

```bash
docker build -t mfn:4.1 .
docker run mfn:4.1
```

GPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Структура

```
mycelium-fractal-net/
├── src/mycelium_fractal_net/
│   ├── __init__.py          # Public API
│   └── model.py             # Core implementation
├── api.py                   # FastAPI server
├── mycelium_fractal_net_v4_1.py  # CLI
├── tests/                   # pytest suite
├── configs/                 # small | medium | large
├── docs/
│   ├── ARCHITECTURE.md
│   ├── MATH_MODEL.md
│   └── ROADMAP.md
├── Dockerfile
└── k8s.yaml
```

---

## Тести

```bash
pytest -q
```

Coverage: Nernst • Turing • STDP • Fractal • Federated • Determinism

---

## Залежності

| Package | Version | Purpose |
|:--------|:--------|:--------|
| torch | ≥2.0.0 | Neural networks |
| numpy | ≥1.24 | Numerical computing |
| sympy | ≥1.12 | Symbolic verification |
| fastapi | ≥0.109.0 | REST API |

---

## Документація

| Документ | Опис |
|:---------|:-----|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Архітектура системи |
| [MATH_MODEL.md](docs/MATH_MODEL.md) | Математична формалізація |
| [ROADMAP.md](docs/ROADMAP.md) | План розвитку |

---

<p align="center">
  <strong>MIT License</strong> · Yaroslav Vasylenko · <a href="https://github.com/neuron7x">@neuron7x</a>
</p>
