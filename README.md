<p align="center">
  <img src="assets/header.svg" alt="Mycelium Fractal Net" width="100%" />
</p>

# MyceliumFractalNet (MFN) v4.1

Біо-інспірована адаптивна нейромережа з фрактальною динамікою, STDP пластичністю та Byzantine-робастним федеративним навчанням.

<p align="center">
  <img src="https://img.shields.io/badge/Version-4.1.0-blue?style=flat-square" alt="Version: 4.1.0" />
  <img src="https://img.shields.io/badge/Python-3.10+-yellow?style=flat-square&logo=python&logoColor=white" alt="Python: 3.10+" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License: MIT" />
  <img src="https://img.shields.io/badge/Tests-100%25_pass-brightgreen?style=flat-square" alt="Tests: 100% pass" />
</p>

<p align="center">
  <img src="assets/morphogenesis.gif" alt="Морфогенез" width="400" />
</p>

-----

## Основні можливості

- **Рівняння Нернста** — обчислення мембранних потенціалів з фізично коректними параметрами
- **Морфогенез Тюрінга** — симуляція росту міцеліальних мереж з активатор-інгібіторною динамікою
- **Фрактальний аналіз** — оцінка розмірності методом box-counting та IFS генерація
- **STDP пластичність** — гетеросинаптичне навчання залежне від часу спайків
- **Sparse Attention** — розріджена увага з top-k селекцією
- **Федеративне навчання** — ієрархічна Krum агрегація з толерантністю до Byzantine атак

-----

## Валідовані метрики

| Метрика | Значення | Опис |
| :--- | :--- | :--- |
| `nernst_E_K` | ≈ -89 мВ | Рівноважний потенціал K⁺ при 37°C |
| `fractal_dim` | 1.4 – 1.9 | Фрактальна розмірність міцеліального поля |
| `lyapunov` | < 0 | Стабільна (контрактивна) динаміка IFS |
| `pot_range` | [-95, 40] мВ | Діапазон мембранних потенціалів |
| `loss_drop` | > 20% | Зменшення loss за тренування |

-----

## Компоненти системи

| Компонент | Функція | Параметри |
| :--- | :--- | :--- |
| `compute_nernst_potential` | Рівняння Нернста | R=8.314, F=96485, T=310K |
| `simulate_mycelium_field` | Дифузійна решітка з Тюрінг-морфогенезом | α=0.18, threshold=0.75 |
| `estimate_fractal_dimension` | Box-counting аналіз | scales=5, min_box=2 |
| `STDPPlasticity` | Spike-Timing Dependent Plasticity | τ±=20ms, A+=0.01, A-=0.012 |
| `SparseAttention` | Top-k розріджена увага | topk=4 |
| `HierarchicalKrumAggregator` | Byzantine-робастна агрегація | clusters=100, f=20% |

-----

## Фізичний фундамент

### Рівняння Нернста

Мембранний потенціал для іона з валентністю z:

$$E = \frac{RT}{zF} \ln\left(\frac{[ion]_{out}}{[ion]_{in}}\right)$$

Для K⁺ при [K]_in=140mM, [K]_out=5mM, T=310K: E_K ≈ -89 мВ

<p align="center">
  <img src="assets/node_dynamics.png" alt="Динаміка вузла" width="600" />
</p>

### Морфогенез Тюрінга

Система активатор-інгібітор:

$$\frac{\partial a}{\partial t} = D_a \nabla^2 a + r_a \cdot a(1-a) - i$$

$$\frac{\partial i}{\partial t} = D_i \nabla^2 i + r_i \cdot (a - i)$$

Параметри: D_a=0.1, D_i=0.05, r_a=0.01, r_i=0.02, threshold=0.75

### Фрактальна топологія

<p align="center">
  <img src="assets/fractal_topology.png" alt="Фрактальна топологія" width="400" />
</p>

Box-counting оцінка розмірності:

$$D = \lim_{\epsilon \to 0} \frac{\ln N(\epsilon)}{\ln(1/\epsilon)}$$

-----

## Встановлення

```bash
git clone https://github.com/neuron7x/mycelium-fractal-net.git
cd mycelium-fractal-net
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Або з dev-залежностями:

```bash
pip install -e ".[dev]"
```

## Використання

### CLI валідація

```bash
python mycelium_fractal_net_v4_1.py --mode validate --seed 42 --epochs 5
```

Приклад виводу:
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

### Python API

```python
from mycelium_fractal_net import (
    compute_nernst_potential,
    simulate_mycelium_field,
    estimate_fractal_dimension,
    run_validation,
)

# Обчислити потенціал Нернста для K⁺
E_K = compute_nernst_potential(
    z_valence=1,
    concentration_out_molar=5e-3,
    concentration_in_molar=140e-3,
    temperature_k=310.0
)
print(f"E_K = {E_K * 1000:.2f} mV")  # ≈ -89 mV

# Симулювати міцеліальне поле
import numpy as np
rng = np.random.default_rng(42)
field, growth_events = simulate_mycelium_field(
    rng=rng,
    grid_size=64,
    steps=64,
    turing_enabled=True
)

# Оцінити фрактальну розмірність
binary = field > -0.060
D = estimate_fractal_dimension(binary)
print(f"Fractal dimension: {D:.4f}")
```

### REST API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Ендпоінти:
| Метод | Шлях | Опис |
|-------|------|------|
| GET | `/health` | Перевірка стану |
| POST | `/validate` | Запуск валідації |
| POST | `/simulate` | Симуляція поля |
| POST | `/nernst` | Обчислення потенціалу Нернста |
| POST | `/federated/aggregate` | Krum агрегація градієнтів |

-----

## Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "mycelium_fractal_net_v4_1.py", "--mode", "validate"]
```

```bash
docker build -t mycelium-fractal-net:v4.1 .
docker run mycelium-fractal-net:v4.1
```

Для GPU підтримки:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

-----

## Структура проекту

```
mycelium-fractal-net/
├── mycelium_fractal_net_v4_1.py   # CLI entrypoint
├── api.py                          # FastAPI сервер
├── src/mycelium_fractal_net/
│   ├── __init__.py
│   └── model.py                    # Основна імплементація
├── tests/                          # Тести (pytest)
├── docs/
│   ├── ARCHITECTURE.md             # Архітектура системи
│   ├── MATH_MODEL.md               # Математична формалізація
│   ├── VALIDATION_NOTES.md         # Очікувані метрики
│   └── ROADMAP.md                  # План розвитку
├── configs/                        # Конфігурації (small/medium/large)
├── examples/                       # Приклади використання
├── assets/                         # Візуальні активи
├── Dockerfile                      # Docker образ
├── k8s.yaml                        # Kubernetes deployment
├── requirements.txt
└── pyproject.toml
```

-----

## Тестування

```bash
pytest -q
```

Тести покривають:
- Фізичну коректність рівняння Нернста (E_K ≈ -89 мВ)
- Box-counting фрактальний аналіз
- Детермінізм (однаковий seed → однаковий результат)
- Стабільність (відсутність NaN/Inf)
- Морфогенез Тюрінга
- STDP пластичність
- Sparse attention
- Федеративну агрегацію

-----

## Залежності

- `torch>=2.0.0` — нейромережа
- `numpy>=1.24` — чисельні обчислення
- `sympy>=1.12` — символьна верифікація
- `fastapi>=0.109.0` — REST API
- `uvicorn>=0.27.0` — ASGI сервер

-----

## Автор

Ярослав Василенко ([@neuron7x](https://github.com/neuron7x))

-----

## Ліцензія

MIT License

-----

<p align="center">
  <a href="docs/ARCHITECTURE.md">Архітектура</a> •
  <a href="docs/MATH_MODEL.md">Математична модель</a> •
  <a href="docs/ROADMAP.md">Roadmap</a>
</p>
