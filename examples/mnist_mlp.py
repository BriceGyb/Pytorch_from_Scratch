"""
MNIST MLP — 784 → 256 → 128 → 10 network trained on MNIST.

Demonstrates micrograd_plus on a real multi-class classification task.
Falls back to synthetic data if scikit-learn is not installed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

try:
    from sklearn.datasets import fetch_openml
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from micrograd import Tensor
from micrograd.nn import Sequential, Linear, ReLU, Dropout
from micrograd.nn.loss import CrossEntropyLoss
from micrograd.optim import Adam


def load_mnist():
    """Load the MNIST dataset. Falls back to synthetic data if sklearn is unavailable."""
    if not HAS_SKLEARN:
        print("sklearn not available — generating synthetic data for demo")
        X = np.random.randn(1000, 784).astype(np.float64)
        y = np.random.randint(0, 10, 1000)
        return X[:800] / 255, y[:800], X[800:] / 255, y[800:]

    print("Loading MNIST from OpenML (this may take a moment on first run)...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype(np.float64) / 255.0
    y = mnist.target.astype(int)

    # MNIST has 70,000 samples: 60,000 train + 10,000 test
    X_train, y_train = X[:60000], y[:60000]
    X_test, y_test = X[60000:], y[60000:]

    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    return X_train, y_train, X_test, y_test


def main():
    print("MNIST MLP Training")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_mnist()
    n = len(X_train)

    # Build the network: 784 → 256 (ReLU, Dropout) → 128 (ReLU) → 10 (logits)
    model = Sequential(
        Linear(784, 256),
        ReLU(),
        Dropout(0.2),
        Linear(256, 128),
        ReLU(),
        Linear(128, 10)
    )

    optimizer = Adam(list(model.parameters()), lr=0.001)
    criterion = CrossEntropyLoss()

    batch_size = 64
    epochs = 10

    print(f"\nModel architecture:\n{model}")
    param_count = sum(p.data.size for p in model.parameters())
    print(f"\nTotal parameters: {param_count:,}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print("-" * 60)

    for epoch in range(epochs):
        model.train()

        # Shuffle training data each epoch
        idx = np.random.permutation(n)
        X_shuffled = X_train[idx]
        y_shuffled = y_train[idx]

        total_loss = 0.0
        correct = 0
        n_batches = 0

        for i in range(0, n, batch_size):
            # Get batch
            Xb = Tensor(X_shuffled[i:i + batch_size])
            yb = y_shuffled[i:i + batch_size]

            # Forward pass
            out = model(Xb)
            loss = criterion(out, yb)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            preds = np.argmax(out.data, axis=1)
            correct += (preds == yb).sum()
            n_batches += 1

        train_acc = correct / n * 100
        avg_loss = total_loss / n_batches

        # Evaluation (no dropout in eval mode)
        model.eval()

        # Run test set in chunks to avoid memory issues
        test_correct = 0
        test_total = len(X_test)
        test_batch = 256

        for j in range(0, test_total, test_batch):
            X_tb = Tensor(X_test[j:j + test_batch])
            out_t = model(X_tb)
            test_preds = np.argmax(out_t.data, axis=1)
            test_correct += (test_preds == y_test[j:j + test_batch]).sum()

        test_acc = test_correct / test_total * 100

        print(f"Epoch {epoch + 1:2d}/{epochs} — "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
