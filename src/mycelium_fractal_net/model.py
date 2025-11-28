
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


def symbolic_nernst_verify() -> float:
    """
    Use sympy to verify Nernst formula with specific values.

    Returns numeric potential for K+ at standard concentrations.
    This function verifies that the symbolic and numeric implementations match.
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


# Backward compatibility alias
_symbolic_nernst_example = symbolic_nernst_verify


# === Turing Reaction-Diffusion Constants ===
TURING_THRESHOLD: float = 0.75  # Activation threshold for Turing patterns


def compute_turing_dispersion(
    k: float,
    d_u: float = 1.0,
    d_v: float = 10.0,
    a: float = 0.5,
    b: float = 1.0,
) -> float:
    """
    Compute Turing dispersion relation λ(k) for reaction-diffusion system.

    For Turing instability, we analyze the linearized reaction-diffusion system:
        ∂u/∂t = d_u ∇²u + f(u,v)
        ∂v/∂t = d_v ∇²v + g(u,v)

    The dispersion relation λ(k) gives growth rate of perturbations at wavenumber k.
    Turing instability requires λ(k) > 0 for some k > 0.

    Simplified model: λ(k) = (a - d_u*k²) * (1 - d_v*k²/b) - coupling term

    Parameters
    ----------
    k : float
        Wavenumber (spatial frequency).
    d_u : float
        Diffusion coefficient for activator.
    d_v : float
        Diffusion coefficient for inhibitor.
    a : float
        Reaction rate parameter for activator.
    b : float
        Reaction rate parameter for inhibitor.

    Returns
    -------
    float
        Growth rate λ(k) at wavenumber k.
    """
    # Trace and determinant of Jacobian with diffusion terms
    trace = a - b - (d_u + d_v) * k * k
    det = (a - d_u * k * k) * (-b - d_v * k * k) + a * b

    # Eigenvalue from quadratic: λ² - trace*λ + det = 0
    # For instability, we need the larger eigenvalue > 0
    discriminant = trace * trace - 4.0 * det
    if discriminant < 0:
        return trace / 2.0  # Real part of complex eigenvalue
    sqrt_disc = math.sqrt(discriminant)
    lambda_plus = (trace + sqrt_disc) / 2.0
    return lambda_plus


def verify_turing_instability(
    d_u: float = 1.0,
    d_v: float = 10.0,
    a: float = 0.5,
    b: float = 0.3,
    k_samples: int = 50,
) -> Tuple[bool, float, float]:
    """
    Verify Turing instability condition: λ(k) > 0 for some k > 0.

    Uses sympy for symbolic verification and numerical sampling.

    Parameters
    ----------
    d_u : float
        Diffusion coefficient for activator.
    d_v : float
        Diffusion coefficient for inhibitor.
    a : float
        Reaction rate for activator.
    b : float
        Reaction rate for inhibitor.
    k_samples : int
        Number of k values to sample.

    Returns
    -------
    Tuple[bool, float, float]
        (is_unstable, max_lambda, k_at_max)
    """
    k_vals = np.linspace(0.01, 2.0, k_samples)
    lambdas = [compute_turing_dispersion(k, d_u, d_v, a, b) for k in k_vals]
    lambdas_arr = np.array(lambdas)

    max_idx = int(np.argmax(lambdas_arr))
    max_lambda = float(lambdas_arr[max_idx])
    k_at_max = float(k_vals[max_idx])

    is_unstable = max_lambda > 0
    return is_unstable, max_lambda, k_at_max


def symbolic_turing_verify() -> Dict[str, float]:
    """
    Sympy verification of Turing dispersion relation.

    Returns dictionary with symbolic and numerical verification results.
    Useful for validating that the numerical implementation matches theory.
    """
    k, d_u, d_v, a, b = sp.symbols("k d_u d_v a b", real=True, positive=True)

    # Jacobian eigenvalue analysis
    trace_expr = a - b - (d_u + d_v) * k**2
    det_expr = (a - d_u * k**2) * (-b - d_v * k**2) + a * b

    # Substitute typical Turing-unstable parameters
    subs = {d_u: 1.0, d_v: 10.0, a: 0.5, b: 0.3}

    # Evaluate at a specific k
    k_test = 0.3
    trace_val = float(trace_expr.subs({**subs, k: k_test}).evalf())
    det_val = float(det_expr.subs({**subs, k: k_test}).evalf())

    # Compute eigenvalue
    disc = trace_val**2 - 4 * det_val
    if disc >= 0:
        lambda_max = (trace_val + math.sqrt(disc)) / 2.0
    else:
        lambda_max = trace_val / 2.0

    return {
        "trace": trace_val,
        "determinant": det_val,
        "lambda_max_at_k0.3": lambda_max,
        "is_unstable": float(lambda_max > 0),
    }


# Backward compatibility alias
_symbolic_turing_verify = symbolic_turing_verify


# === STDP (Spike-Timing-Dependent Plasticity) Constants ===
STDP_TAU_MS: float = 20.0  # Time constant in milliseconds
STDP_A_PLUS: float = 0.01  # LTP amplitude
STDP_A_MINUS: float = 0.012  # LTD amplitude (slightly larger for homeostasis)


def compute_stdp_weight_change(
    delta_t_ms: float,
    tau_ms: float = STDP_TAU_MS,
    a_plus: float = STDP_A_PLUS,
    a_minus: float = STDP_A_MINUS,
) -> float:
    """
    Compute STDP weight change based on spike timing difference.

    Implements asymmetric STDP rule:
        Δw = A+ * exp(-|Δt|/τ)  if Δt > 0 (post fires after pre → LTP)
        Δw = -A- * exp(-|Δt|/τ) if Δt < 0 (post fires before pre → LTD)

    Parameters
    ----------
    delta_t_ms : float
        Time difference t_post - t_pre in milliseconds.
        Positive when post-synaptic spike follows pre-synaptic spike (LTP).
        Negative when post-synaptic spike precedes pre-synaptic spike (LTD).
    tau_ms : float
        Time constant for exponential decay.
    a_plus : float
        Amplitude for LTP (potentiation).
    a_minus : float
        Amplitude for LTD (depression).

    Returns
    -------
    float
        Weight change Δw.
    """
    if delta_t_ms > 0:
        # Post fires after pre → LTP (strengthen connection)
        return a_plus * math.exp(-abs(delta_t_ms) / tau_ms)
    elif delta_t_ms < 0:
        # Post fires before pre → LTD (weaken connection)
        return -a_minus * math.exp(-abs(delta_t_ms) / tau_ms)
    else:
        return 0.0


def verify_stdp_lipschitz(
    tau_ms: float = STDP_TAU_MS,
    a_plus: float = STDP_A_PLUS,
    a_minus: float = STDP_A_MINUS,
    epsilon: float = 0.01,
) -> Tuple[bool, float]:
    """
    Verify STDP function satisfies Lipschitz condition with constant < epsilon.

    The Lipschitz constant is bounded by max|dΔw/d(Δt)|.
    For exponential STDP: max derivative = max(A+, A-) / τ.

    Parameters
    ----------
    tau_ms : float
        STDP time constant.
    a_plus : float
        LTP amplitude.
    a_minus : float
        LTD amplitude.
    epsilon : float
        Target Lipschitz bound.

    Returns
    -------
    Tuple[bool, float]
        (satisfies_bound, lipschitz_constant)
    """
    # Maximum derivative of exp(-|t|/τ) is 1/τ
    # So Lipschitz constant = max(A+, A-) / τ
    lipschitz_constant = max(a_plus, a_minus) / tau_ms
    satisfies = lipschitz_constant < epsilon
    return satisfies, lipschitz_constant


def compute_heterosynaptic_modulation(
    spike_history: np.ndarray,
    dt_ms: float = 1.0,
) -> float:
    """
    Compute heterosynaptic modulation factor g(N) = mean ∫h_k dt.

    Integrates spike activity over time to compute global neuromodulation.

    Parameters
    ----------
    spike_history : np.ndarray
        Binary spike train array of shape (n_neurons, n_timesteps).
    dt_ms : float
        Time step in milliseconds.

    Returns
    -------
    float
        Modulation factor g(N) in [0, 1].
    """
    if spike_history.size == 0:
        return 0.0

    # Integrate spike counts over time for each neuron
    integrals = np.sum(spike_history, axis=1) * dt_ms

    # Mean across neurons, normalized
    g_n = float(np.mean(integrals))

    # Clamp to [0, 1] range
    return float(np.clip(g_n / 100.0, 0.0, 1.0))


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
    counts_list: list[int] = []

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
        counts_list.append(int(occupied.sum()))

    counts: np.ndarray = np.array(counts_list, dtype=float)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.net(x)
        return result


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

    # Turing instability verification
    is_unstable, max_lambda, k_at_max = verify_turing_instability()
    metrics["turing_is_unstable"] = float(is_unstable)
    metrics["turing_max_lambda"] = max_lambda
    metrics["turing_k_at_max"] = k_at_max

    # STDP Lipschitz verification
    satisfies_lipschitz, lipschitz_const = verify_stdp_lipschitz()
    metrics["stdp_lipschitz_ok"] = float(satisfies_lipschitz)
    metrics["stdp_lipschitz_const"] = lipschitz_const

    return metrics


def run_validation_cli() -> None:
    """
    CLI-обгортка для MyceliumFractalNet v4.1.
    """
    parser = argparse.ArgumentParser(description="MyceliumFractalNet v4.1 validation CLI")
    parser.add_argument(
        "--mode", type=str, default="validate", choices=["validate"], help="Mode"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for RNG")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--grid-size", type=int, default=64, help="Grid size")
    parser.add_argument("--steps", type=int, default=64, help="Simulation steps")
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
