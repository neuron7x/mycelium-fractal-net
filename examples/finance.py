#!/usr/bin/env python3
"""
Finance Example - MyceliumFractalNet v4.1

This example demonstrates using MyceliumFractalNet for financial time series
prediction. The network's adaptive fractal structure and bio-inspired dynamics
are well-suited for capturing complex patterns in financial data.

Features demonstrated:
- Loading and preprocessing financial data
- Creating a MyceliumFractalNet model for regression
- Training with STDP-inspired learning
- Evaluating prediction performance
"""

import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from mfn import MyceliumFractalNet, load_config


def generate_synthetic_financial_data(
    n_samples: int = 1000,
    sequence_length: int = 32,
    n_features: int = 5,
    noise_level: float = 0.1,
) -> tuple:
    """Generate synthetic financial time series data.

    Creates artificial stock-like data with trends, seasonality, and noise.

    Args:
        n_samples: Number of samples to generate.
        sequence_length: Length of each sequence.
        n_features: Number of features (e.g., OHLCV).
        noise_level: Standard deviation of noise.

    Returns:
        Tuple of (features, targets) as numpy arrays.
    """
    np.random.seed(42)

    # Generate base price series with trend and seasonality
    t = np.linspace(0, 4 * np.pi, n_samples + sequence_length)

    # Multiple patterns: trend, daily cycle, weekly cycle
    trend = 0.01 * np.arange(len(t))
    daily_cycle = 0.1 * np.sin(t * 5)
    weekly_cycle = 0.2 * np.sin(t)
    base_price = 100 + trend + daily_cycle + weekly_cycle

    # Add random walk component
    random_walk = np.cumsum(np.random.randn(len(t)) * 0.5)
    price = base_price + random_walk

    # Generate OHLCV-like features
    features = np.zeros((n_samples, n_features))
    targets = np.zeros(n_samples)

    for i in range(n_samples):
        window = price[i:i + sequence_length]

        # Feature engineering
        features[i, 0] = window[-1]  # Close price
        features[i, 1] = window.max()  # High
        features[i, 2] = window.min()  # Low
        features[i, 3] = window.mean()  # Average
        if n_features > 4:
            features[i, 4] = window.std()  # Volatility

        # Target: next price change (normalized)
        targets[i] = (price[i + sequence_length] - window[-1]) / window[-1]

    # Normalize features
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    # Add noise
    features += np.random.randn(*features.shape) * noise_level

    return features, targets


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 0.001,
) -> dict:
    """Train the model on financial data.

    Args:
        model: MyceliumFractalNet model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.

    Returns:
        Dictionary containing training history.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "val_loss": []}

    print("\nTraining MyceliumFractalNet on financial data...")
    print("-" * 50)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # Apply STDP update
            model.apply_stdp_update(learning_rate=learning_rate * 0.1)

        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)

        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        history["val_loss"].append(avg_val_loss)

        print(f"Epoch {epoch + 1:3d}/{epochs}: "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {avg_val_loss:.6f}")

    return history


def evaluate_model(model: nn.Module, test_loader: DataLoader) -> dict:
    """Evaluate model performance.

    Args:
        model: Trained model.
        test_loader: Test data loader.

    Returns:
        Dictionary containing evaluation metrics.
    """
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x).squeeze()
            predictions.extend(outputs.numpy())
            actuals.extend(batch_y.numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))

    # Direction accuracy (predicting up/down correctly)
    direction_correct = np.mean(np.sign(predictions) == np.sign(actuals))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "direction_accuracy": direction_correct,
    }


def main():
    """Main function to run the finance example."""
    print("=" * 60)
    print("MyceliumFractalNet v4.1 - Finance Example")
    print("=" * 60)

    # Configuration
    config = {
        "architecture": {
            "input_dim": 5,  # OHLCV features
            "hidden_dim": 64,
            "output_dim": 1,  # Price change prediction
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
            "tau_plus": 20.0,
            "tau_minus": 20.0,
        },
    }

    print("\n[1/5] Generating synthetic financial data...")
    features, targets = generate_synthetic_financial_data(
        n_samples=1000,
        n_features=5,
    )
    print(f"      Generated {len(features)} samples with {features.shape[1]} features")

    print("\n[2/5] Preparing data loaders...")
    # Split data
    train_size = int(0.7 * len(features))
    val_size = int(0.15 * len(features))

    X_train = torch.FloatTensor(features[:train_size])
    y_train = torch.FloatTensor(targets[:train_size])
    X_val = torch.FloatTensor(features[train_size:train_size + val_size])
    y_val = torch.FloatTensor(targets[train_size:train_size + val_size])
    X_test = torch.FloatTensor(features[train_size + val_size:])
    y_test = torch.FloatTensor(targets[train_size + val_size:])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=32, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=32, shuffle=False
    )

    print(f"      Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    print("\n[3/5] Creating MyceliumFractalNet model...")
    model = MyceliumFractalNet(config)
    stats = model.get_network_stats()
    print(f"      Parameters: {stats['num_parameters']:,}")

    print("\n[4/5] Training model...")
    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=5,
        learning_rate=0.001,
    )

    print("\n[5/5] Evaluating model...")
    metrics = evaluate_model(model, test_loader)
    print("-" * 50)
    print(f"      RMSE: {metrics['rmse']:.6f}")
    print(f"      MAE: {metrics['mae']:.6f}")
    print(f"      Direction Accuracy: {metrics['direction_accuracy']:.2%}")

    print("\n" + "=" * 60)
    print("Finance example completed successfully!")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    main()
