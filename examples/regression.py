"""
Linear Regression Example — y = 2x + 1 + noise

Demonstrates that micrograd_plus can fit a simple linear model using
gradient descent with MSELoss and the SGD optimizer.

The true underlying function is: y = 2.0 * x + 1.0
We add Gaussian noise to simulate real-world data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from micrograd import Tensor
from micrograd.nn import Linear, Sequential
from micrograd.nn.loss import MSELoss
from micrograd.optim import SGD, Adam


def generate_data(n_samples=100, noise_std=0.3, seed=42):
    """Generate synthetic linear regression data: y = 2x + 1 + noise."""
    np.random.seed(seed)
    X = np.random.uniform(-3, 3, (n_samples, 1))
    y = 2.0 * X + 1.0 + np.random.randn(n_samples, 1) * noise_std
    return X, y


def main():
    print("Linear Regression: y = 2x + 1 + noise")
    print("=" * 50)

    # Generate synthetic data
    X_np, y_np = generate_data(n_samples=200, noise_std=0.5)

    # Train/test split
    split = 160
    X_train, y_train = X_np[:split], y_np[:split]
    X_test, y_test = X_np[split:], y_np[split:]

    # Convert to Tensors
    X_train_t = Tensor(X_train, requires_grad=False)
    y_train_t = Tensor(y_train, requires_grad=False)
    X_test_t = Tensor(X_test, requires_grad=False)
    y_test_t = Tensor(y_test, requires_grad=False)

    # Simple linear model: one Linear layer with 1 input and 1 output
    # This will learn weights ~ [2.0] and bias ~ [1.0]
    model = Linear(1, 1, bias=True)
    optimizer = SGD(list(model.parameters()), lr=0.01, momentum=0.9)
    criterion = MSELoss()

    print(f"\nTraining on {split} samples, evaluating on {len(X_test)} samples")
    print(f"True parameters: weight=2.0, bias=1.0")
    print("-" * 50)

    # Training loop
    epochs = 500
    for epoch in range(epochs):
        # Forward
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            # Compute test loss (no grad needed)
            model.eval()
            test_pred = model(X_test_t)
            test_loss = criterion(test_pred, y_test_t)
            model.train()
            print(f"Epoch {epoch + 1:4d}/{epochs} — "
                  f"Train Loss: {loss.item():.6f} | "
                  f"Test Loss: {test_loss.item():.6f}")

    # Show learned parameters
    print("\nLearned parameters:")
    print(f"  weight = {model.weight.data[0, 0]:.4f}  (true: 2.0)")
    print(f"  bias   = {model.bias.data[0]:.4f}  (true: 1.0)")

    # Final predictions on a few test points
    print("\nSample predictions:")
    print("-" * 50)
    model.eval()
    test_preds = model(X_test_t)
    for i in range(min(8, len(X_test))):
        x_val = X_test[i, 0]
        y_true = y_test[i, 0]
        y_pred = test_preds.data[i, 0]
        print(f"  x={x_val:+.2f} -> pred={y_pred:+.4f}, true={y_true:+.4f}")

    # Compute R² score manually
    ss_res = np.sum((y_test - test_preds.data) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    print(f"\nR² score on test set: {r2:.4f}")


def demo_polynomial_regression():
    """
    Extension: polynomial regression y = x^2 - 2x + 1 using feature expansion.
    Demonstrates using multiple features.
    """
    print("\n" + "=" * 50)
    print("Polynomial Regression: y = x^2 - 2x + 1")
    print("=" * 50)

    np.random.seed(0)
    x = np.linspace(-3, 3, 200)
    y_true = x ** 2 - 2 * x + 1
    y_noisy = y_true + np.random.randn(200) * 0.3

    # Feature matrix: [x, x^2]
    X = np.column_stack([x, x ** 2])
    y = y_noisy.reshape(-1, 1)

    # Split
    X_train, y_train = X[:160], y[:160]
    X_test, y_test = X[160:], y[160:]

    X_train_t = Tensor(X_train)
    y_train_t = Tensor(y_train)
    X_test_t = Tensor(X_test)
    y_test_t = Tensor(y_test)

    # Linear model over [x, x^2] features
    # Should learn weights close to [-2, 1] and bias close to 1
    model = Linear(2, 1, bias=True)
    optimizer = Adam(list(model.parameters()), lr=0.01)
    criterion = MSELoss()

    for epoch in range(1000):
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    test_pred = model(X_test_t)
    test_loss = criterion(test_pred, y_test_t)

    print(f"Final test MSE: {test_loss.item():.6f}")
    print(f"Learned weights: {model.weight.data[0]}  (true: [-2.0, 1.0])")
    print(f"Learned bias: {model.bias.data[0]:.4f}  (true: 1.0)")


if __name__ == "__main__":
    main()
    demo_polynomial_regression()
