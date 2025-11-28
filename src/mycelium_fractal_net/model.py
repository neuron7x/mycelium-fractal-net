
"""
Core implementation of MyceliumFractalNet v4.1.

Цей модуль містить:
- compute_nernst_potential: фізично коректне рівняння Нернста;
- simulate_mycelium_field: дифузійна решітка з "міцеліальними" подіями;
- estimate_fractal_dimension: box-counting оцінка фрактальної розмірності;
- MyceliumFractalNet: невелика NN-модель над статистиками поля;
- run_validation / run_validation_cli: інтегрований валідаційний цикл.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import sympy as sp
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# === Фізичні константи (SI) ===
R_GAS_CONSTANT: float = 8.314  # J/(mol*K)
FARADAY_CONSTANT: float = 96485.33212  # C/mol
BODY_TEMPERATURE_K: float = 310.0  # K (~37°C)


def compute_nernst_potential(
    z_valence: int,
    concentration_out_molar: float,
    concentration_in_molar: float,
    temperature_k: float = BODY_TEMPERATURE_K,
) -> float:
    """
    Обчислити різницю потенціалу за рівнянням Нернста (вольти).

    E = (R*T)/(z*F) * ln([ion]_out / [ion]_in)

    Параметри
    ---------
    z_valence : int
        Валентність іона (для K+ = 1).
    concentration_out_molar : float
        Зовнішня концентрація (моль/л).
    concentration_in_molar : float
        Внутрішня концентрація (моль/л).
    temperature_k : float
        Температура в Кельвінах.

    Повертає
    --------
    float
        Мембранний потенціал у вольтах.
    """
    if concentration_out_molar <= 0 or concentration_in_molar <= 0:
        raise ValueError("Concentrations must be positive for Nernst potential.")

    ratio = concentration_out_molar / concentration_in_molar
    return (R_GAS_CONSTANT * temperature_k) / (z_valence * FARADAY_CONSTANT) * math.log(ratio)


def _symbolic_nernst_example() -> float:
    """
    Використовує sympy для підтвердження формули Нернста на конкретних значеннях.

    Повертає числове значення потенціалу для K+ при стандартних концентраціях.
    """
    R, T, z, F, c_out, c_in = sp.symbols("R T z F c_out c_in", positive=True)
    E_expr = (R * T) / (z * F) * sp.log(c_out / c_in)

    subs = {
        R: R_GAS_CONSTANT,
        T: BODY_TEMPERATURE_K,
        z: 1,
        F: FARADAY_CONSTANT,
        c_out: 5e-3,
        c_in: 140e-3,
    }
    E_val = float(E_expr.subs(subs).evalf())
    return E_val


def simulate_mycelium_field(
    rng: np.random.Generator,
    grid_size: int = 64,
    steps: int = 64,
    alpha: float = 0.18,
    spike_probability: float = 0.25,
) -> Tuple[np.ndarray, int]:
    """
    Симуляція "міцеліального" поля потенціалів на 2D-решітці.

    Модель:
    - поле V ініціалізується навколо -70 мВ;
    - кожен крок: іноді додаються локальні "спайки" (події росту);
    - потім застосовується дискретний лапласіан (дифузія);
    - значення обмежуються фізично правдоподібним діапазоном.

    Параметри
    ---------
    rng : np.random.Generator
        Генератор випадкових чисел (для детермінізму в тестах).
    grid_size : int
        Розмір поля N x N.
    steps : int
        Кількість кроків симуляції.
    alpha : float
        Коефіцієнт дифузії.
    spike_probability : float
        Імовірність події росту на кроці.

    Повертає
    --------
    field : np.ndarray
        Масив форми (N, N) у вольтах.
    growth_events : int
        Кількість подій росту.
    """
    # Початковий стан ~ -70 мВ
    field = rng.normal(loc=-0.07, scale=0.005, size=(grid_size, grid_size))
    growth_events = 0

    for _ in range(steps):
        if rng.random() < spike_probability:
            i = int(rng.integers(0, grid_size))
            j = int(rng.integers(0, grid_size))
            field[i, j] += float(rng.normal(loc=0.02, scale=0.005))
            growth_events += 1

        # Лапласіан через зсуви
        up = np.roll(field, 1, axis=0)
        down = np.roll(field, -1, axis=0)
        left = np.roll(field, 1, axis=1)
        right = np.roll(field, -1, axis=1)
        laplacian = up + down + left + right - 4.0 * field
        field = field + alpha * laplacian

        # Обмеження діапазону (≈ [-95, 40] мВ)
        field = np.clip(field, -0.095, 0.040)

    return field, growth_events


def estimate_fractal_dimension(
    binary_field: np.ndarray,
    min_box_size: int = 2,
    max_box_size: int | None = None,
    num_scales: int = 5,
) -> float:
    """
    Box-counting оцінка фрактальної розмірності бінарного поля.

    Параметри
    ---------
    binary_field : np.ndarray
        Масив 0/1 (False/True) форми (N, N).
    min_box_size : int
        Мінімальний розмір "коробки".
    max_box_size : int | None
        Максимальний розмір коробки (якщо None — до N//2).
    num_scales : int
        Кількість масштабів між min та max (логарифмічно).

    Повертає
    --------
    float
        Оцінка фрактальної розмірності.
    """
    if binary_field.ndim != 2 or binary_field.shape[0] != binary_field.shape[1]:
        raise ValueError("binary_field must be a square 2D array.")

    n = binary_field.shape[0]
    if max_box_size is None:
        max_box_size = max(min_box_size * (2 ** (num_scales - 1)), min_box_size)
        max_box_size = min(max_box_size, n // 2 if n >= 4 else n)

    if max_box_size < min_box_size:
        max_box_size = min_box_size

    sizes = np.geomspace(min_box_size, max_box_size, num_scales).astype(int)
    sizes = np.unique(sizes)
    counts = []

    for size in sizes:
        if size <= 0:
            continue
        n_boxes = n // size
        if n_boxes == 0:
            continue
        reshaped = binary_field[: n_boxes * size, : n_boxes * size].reshape(
            n_boxes, size, n_boxes, size
        )
        occupied = reshaped.any(axis=(1, 3))
        counts.append(occupied.sum())

    counts = np.array(counts, dtype=float)
    valid = counts > 0
    if valid.sum() < 2:
        return 0.0

    sizes = sizes[valid]
    counts = counts[valid]

    inv_eps = 1.0 / sizes.astype(float)
    log_inv_eps = np.log(inv_eps)
    log_counts = np.log(counts)

    coeffs = np.polyfit(log_inv_eps, log_counts, 1)
    fractal_dim = float(coeffs[0])
    return fractal_dim


class MyceliumFractalNet(nn.Module):
    """
    Проста NN-модель, яка приймає статистики поля та передбачає скалярну ціль.

    Вона не претендує на глибинну біологічну реалістичність, але демонструє
    повний ML-пайплайн (forward, loss, train, валідація).
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


@dataclass
class ValidationConfig:
    seed: int = 42
    epochs: int = 1
    batch_size: int = 4
    grid_size: int = 64
    steps: int = 64
    device: str = "cpu"


def _build_dataset(cfg: ValidationConfig) -> Tuple[TensorDataset, Dict[str, float]]:
    """
    Побудувати невеликий датасет зі статистик поля та таргетом.
    """
    rng = np.random.default_rng(cfg.seed)

    num_samples = 16
    fields = []
    stats = []

    for _ in range(num_samples):
        field, _ = simulate_mycelium_field(rng, grid_size=cfg.grid_size, steps=cfg.steps)
        fields.append(field)
        binary = field > -0.060  # -60 мВ
        D = estimate_fractal_dimension(binary)
        mean_pot = float(field.mean())
        std_pot = float(field.std())
        max_pot = float(field.max())
        stats.append((D, mean_pot, std_pot, max_pot))

    stats_arr = np.asarray(stats, dtype=np.float32)
    # Нормування діапазону потенціалів (в Вольтах) до ~[-1, 1]
    stats_arr[:, 1:] *= 100.0

    # Таргет: проста лінійна комбінація статистик
    targets = (0.5 * stats_arr[:, 0] + 0.2 * stats_arr[:, 1] - 0.1 * stats_arr[:, 2]).reshape(-1, 1)

    x_tensor = torch.from_numpy(stats_arr)
    y_tensor = torch.from_numpy(targets.astype(np.float32))
    dataset = TensorDataset(x_tensor, y_tensor)

    # Глобальні метрики по полях
    all_field = np.stack(fields, axis=0)
    meta = {
        "pot_min_mV": float(all_field.min() * 1000.0),
        "pot_max_mV": float(all_field.max() * 1000.0),
    }

    return dataset, meta


def run_validation(cfg: ValidationConfig | None = None) -> Dict[str, float]:
    """
    Запустити повний валідаційний цикл: симуляція + NN train + метрики.

    Повертає словник з ключами:
    - loss_start, loss_final
    - pot_min_mV, pot_max_mV
    - example_fractal_dim
    """
    if cfg is None:
        cfg = ValidationConfig()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    dataset, meta = _build_dataset(cfg)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    device = torch.device(cfg.device)
    model = MyceliumFractalNet().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    loss_start: float | None = None
    loss_final: float = float("nan")

    for _ in range(cfg.epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimiser.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimiser.step()

            val = float(loss.item())
            if loss_start is None:
                loss_start = val
            loss_final = val

    if loss_start is None:
        loss_start = loss_final

    # Фрактальна розмірність для одного поля (репрезентативна)
    rng = np.random.default_rng(cfg.seed + 1)
    field, growth_events = simulate_mycelium_field(rng, grid_size=cfg.grid_size, steps=cfg.steps)
    binary = field > -0.060
    D = estimate_fractal_dimension(binary)

    metrics: Dict[str, float] = {
        "loss_start": float(loss_start),
        "loss_final": float(loss_final),
        "loss_drop": float(loss_start - loss_final),
        "pot_min_mV": meta["pot_min_mV"],
        "pot_max_mV": meta["pot_max_mV"],
        "example_fractal_dim": float(D),
        "growth_events": float(growth_events),
    }

    # Перевірка sympy-варіанту Нернста (не використовується в loss, але валідує формулу)
    E_symbolic = _symbolic_nernst_example()
    E_numeric = compute_nernst_potential(1, 5e-3, 140e-3)
    metrics["nernst_symbolic_mV"] = float(E_symbolic * 1000.0)
    metrics["nernst_numeric_mV"] = float(E_numeric * 1000.0)

    return metrics


def run_validation_cli() -> None:
    """
    CLI-обгортка для MyceliumFractalNet v4.1.
    """
    parser = argparse.ArgumentParser(description="MyceliumFractalNet v4.1 validation CLI")
    parser.add_argument("--mode", type=str, default="validate", choices=["validate"], help="Режим роботи")
    parser.add_argument("--seed", type=int, default=42, help="Seed для RNG")
    parser.add_argument("--epochs", type=int, default=1, help="Кількість епох")
    parser.add_argument("--batch-size", type=int, default=4, help="Розмір батча")
    parser.add_argument("--grid-size", type=int, default=64, help="Розмір решітки")
    parser.add_argument("--steps", type=int, default=64, help="Кроки симуляції")
    args = parser.parse_args()

    cfg = ValidationConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grid_size=args.grid_size,
        steps=args.steps,
    )

    metrics = run_validation(cfg)

    print("=== MyceliumFractalNet v4.1 :: validation ===")
    for k, v in metrics.items():
        print(f"{k:24s}: {v: .6f}")
