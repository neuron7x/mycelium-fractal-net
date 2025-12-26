"""Performance benchmarks for MyceliumFractalNet.

This module provides performance regression tests to prevent
performance degradation across versions.

Benchmarks include:
- Forward pass latency
- Morphogenesis growth operation latency
- Memory usage during training
- Throughput measurements

Run with: python benchmarks/benchmark_core.py
"""

import json
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mycelium_fractal_net import estimate_fractal_dimension, simulate_mycelium_field
from mycelium_fractal_net.model import MyceliumFractalNet


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    name: str
    metric_value: float
    metric_unit: str
    target_value: float
    passed: bool
    timestamp: str


class BenchmarkSuite:
    """Performance benchmarks to prevent regressions."""

    def __init__(self, results_dir: Optional[Path] = None) -> None:
        """Initialize benchmark suite.

        Args:
            results_dir: Directory to save results (default: benchmarks/results)
        """
        self.results: list[BenchmarkResult] = []
        self.results_dir = results_dir or Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)

    def benchmark_forward_pass(self) -> BenchmarkResult:
        """Measure forward pass latency.

        Target: <10ms for batch_size=32, input_dim=128, hidden_dim=64
        """
        torch.manual_seed(42)

        model = MyceliumFractalNet(input_dim=128, hidden_dim=64)
        model.eval()

        x = torch.randn(32, 128)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)

        # Benchmark
        start = time.perf_counter()
        num_iterations = 100
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(x)
        end = time.perf_counter()

        avg_latency_ms = (end - start) / num_iterations * 1000
        target_ms = 10.0

        result = BenchmarkResult(
            name="forward_pass_latency",
            metric_value=avg_latency_ms,
            metric_unit="ms",
            target_value=target_ms,
            passed=avg_latency_ms < target_ms,
            timestamp=datetime.now().isoformat(),
        )

        print(f"Forward pass latency: {avg_latency_ms:.2f} ms (target: <{target_ms} ms)")
        if not result.passed:
            print(f"  WARNING: Latency {avg_latency_ms:.2f}ms exceeds {target_ms}ms target")

        self.results.append(result)
        return result

    def benchmark_forward_pass_large_batch(self) -> BenchmarkResult:
        """Measure forward pass latency with larger batch.

        Target: <50ms for batch_size=128
        """
        torch.manual_seed(42)

        model = MyceliumFractalNet(input_dim=64, hidden_dim=64)
        model.eval()

        x = torch.randn(128, 64)

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(x)

        # Benchmark
        start = time.perf_counter()
        num_iterations = 50
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(x)
        end = time.perf_counter()

        avg_latency_ms = (end - start) / num_iterations * 1000
        target_ms = 50.0

        result = BenchmarkResult(
            name="forward_pass_large_batch",
            metric_value=avg_latency_ms,
            metric_unit="ms",
            target_value=target_ms,
            passed=avg_latency_ms < target_ms,
            timestamp=datetime.now().isoformat(),
        )

        print(f"Forward pass (batch=128): {avg_latency_ms:.2f} ms (target: <{target_ms} ms)")

        self.results.append(result)
        return result

    def benchmark_field_simulation(self) -> BenchmarkResult:
        """Measure field simulation latency.

        Target: <100ms for 64x64 grid, 100 steps
        """
        rng = np.random.default_rng(42)

        # Warmup
        for _ in range(3):
            _, _ = simulate_mycelium_field(rng, grid_size=32, steps=20)

        # Benchmark
        rng = np.random.default_rng(42)
        start = time.perf_counter()
        num_iterations = 10
        for _ in range(num_iterations):
            rng = np.random.default_rng(42)
            _, _ = simulate_mycelium_field(rng, grid_size=64, steps=100, turing_enabled=True)
        end = time.perf_counter()

        avg_latency_ms = (end - start) / num_iterations * 1000
        target_ms = 100.0

        result = BenchmarkResult(
            name="field_simulation",
            metric_value=avg_latency_ms,
            metric_unit="ms",
            target_value=target_ms,
            passed=avg_latency_ms < target_ms,
            timestamp=datetime.now().isoformat(),
        )

        print(
            f"Field simulation (64x64, 100 steps): {avg_latency_ms:.2f} ms "
            f"(target: <{target_ms} ms)"
        )

        self.results.append(result)
        return result

    def benchmark_fractal_dimension(self) -> BenchmarkResult:
        """Measure fractal dimension estimation latency.

        Target: <50ms for 64x64 binary field
        """
        rng = np.random.default_rng(42)
        binary_field = rng.random((64, 64)) > 0.5

        # Warmup
        for _ in range(5):
            _ = estimate_fractal_dimension(binary_field)

        # Benchmark
        start = time.perf_counter()
        num_iterations = 50
        for _ in range(num_iterations):
            _ = estimate_fractal_dimension(binary_field)
        end = time.perf_counter()

        avg_latency_ms = (end - start) / num_iterations * 1000
        target_ms = 50.0

        result = BenchmarkResult(
            name="fractal_dimension",
            metric_value=avg_latency_ms,
            metric_unit="ms",
            target_value=target_ms,
            passed=avg_latency_ms < target_ms,
            timestamp=datetime.now().isoformat(),
        )

        print(f"Fractal dimension estimation: {avg_latency_ms:.2f} ms (target: <{target_ms} ms)")

        self.results.append(result)
        return result

    def benchmark_training_step(self) -> BenchmarkResult:
        """Measure single training step latency.

        Target: <20ms for batch_size=32
        """
        torch.manual_seed(42)

        model = MyceliumFractalNet(input_dim=64, hidden_dim=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()

        x = torch.randn(32, 64)
        y = torch.randn(32, 1)

        # Warmup
        for _ in range(5):
            _ = model.train_step(x, y, optimizer, criterion)

        # Benchmark
        start = time.perf_counter()
        num_iterations = 50
        for _ in range(num_iterations):
            _ = model.train_step(x, y, optimizer, criterion)
        end = time.perf_counter()

        avg_latency_ms = (end - start) / num_iterations * 1000
        target_ms = 20.0

        result = BenchmarkResult(
            name="training_step",
            metric_value=avg_latency_ms,
            metric_unit="ms",
            target_value=target_ms,
            passed=avg_latency_ms < target_ms,
            timestamp=datetime.now().isoformat(),
        )

        print(f"Training step: {avg_latency_ms:.2f} ms (target: <{target_ms} ms)")

        self.results.append(result)
        return result

    def benchmark_memory_usage(self) -> BenchmarkResult:
        """Measure peak memory usage during training.

        Target: <500MB for standard model
        """
        tracemalloc.start()

        torch.manual_seed(42)

        model = MyceliumFractalNet(input_dim=64, hidden_dim=128)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()

        x = torch.randn(64, 64)
        y = torch.randn(64, 1)

        # Run training loop
        for _ in range(100):
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        target_mb = 500.0

        result = BenchmarkResult(
            name="memory_usage",
            metric_value=peak_mb,
            metric_unit="MB",
            target_value=target_mb,
            passed=peak_mb < target_mb,
            timestamp=datetime.now().isoformat(),
        )

        print(f"Peak memory usage: {peak_mb:.2f} MB (target: <{target_mb} MB)")

        self.results.append(result)
        return result

    def benchmark_throughput(self) -> BenchmarkResult:
        """Measure inference throughput.

        Target: >1000 samples/second
        """
        torch.manual_seed(42)

        model = MyceliumFractalNet(input_dim=64, hidden_dim=64)
        model.eval()

        batch_size = 64
        x = torch.randn(batch_size, 64)

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(x)

        # Benchmark
        num_batches = 100
        start = time.perf_counter()
        for _ in range(num_batches):
            with torch.no_grad():
                _ = model(x)
        end = time.perf_counter()

        total_samples = num_batches * batch_size
        elapsed = end - start
        throughput = total_samples / elapsed

        target_throughput = 1000.0

        result = BenchmarkResult(
            name="throughput",
            metric_value=throughput,
            metric_unit="samples/sec",
            target_value=target_throughput,
            passed=throughput > target_throughput,
            timestamp=datetime.now().isoformat(),
        )

        print(
            f"Inference throughput: {throughput:.0f} samples/sec "
            f"(target: >{target_throughput} samples/sec)"
        )

        self.results.append(result)
        return result

    def benchmark_model_initialization(self) -> BenchmarkResult:
        """Measure model initialization time.

        Target: <100ms
        """
        torch.manual_seed(42)

        # Warmup
        for _ in range(3):
            _ = MyceliumFractalNet(input_dim=64, hidden_dim=64)

        # Benchmark
        start = time.perf_counter()
        num_iterations = 20
        for _ in range(num_iterations):
            _ = MyceliumFractalNet(input_dim=64, hidden_dim=64)
        end = time.perf_counter()

        avg_latency_ms = (end - start) / num_iterations * 1000
        target_ms = 100.0

        result = BenchmarkResult(
            name="model_initialization",
            metric_value=avg_latency_ms,
            metric_unit="ms",
            target_value=target_ms,
            passed=avg_latency_ms < target_ms,
            timestamp=datetime.now().isoformat(),
        )

        print(f"Model initialization: {avg_latency_ms:.2f} ms (target: <{target_ms} ms)")

        self.results.append(result)
        return result

    def run_all(self) -> list[BenchmarkResult]:
        """Run all benchmarks and return results."""
        print("\n" + "=" * 60)
        print("MyceliumFractalNet Performance Benchmarks")
        print("=" * 60 + "\n")

        self.benchmark_forward_pass()
        self.benchmark_forward_pass_large_batch()
        self.benchmark_field_simulation()
        self.benchmark_fractal_dimension()
        self.benchmark_training_step()
        self.benchmark_memory_usage()
        self.benchmark_throughput()
        self.benchmark_model_initialization()

        # Summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        print(f"\nPassed: {passed}/{total}")

        if passed < total:
            print("\nFailed benchmarks:")
            for r in self.results:
                if not r.passed:
                    print(
                        f"  - {r.name}: {r.metric_value:.2f} {r.metric_unit} "
                        f"(target: {r.target_value} {r.metric_unit})"
                    )

        return self.results

    def save_results(self, filename: Optional[str] = None) -> Path:
        """Save benchmark results to JSON file.

        Args:
            filename: Output filename (default: benchmark_TIMESTAMP.json)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{timestamp}.json"

        output_path = self.results_dir / filename

        results_dict = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": [asdict(r) for r in self.results],
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "failed": sum(1 for r in self.results if not r.passed),
            },
        }

        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nResults saved to: {output_path}")
        return output_path


def run_benchmarks() -> int:
    """Run all benchmarks and return exit code.

    Returns:
        0 if all benchmarks pass, 1 otherwise
    """
    suite = BenchmarkSuite()
    results = suite.run_all()
    suite.save_results()

    # Return non-zero if any benchmark failed
    all_passed = all(r.passed for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = run_benchmarks()
    sys.exit(exit_code)
