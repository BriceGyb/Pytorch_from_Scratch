"""
XOR Problem — Demo that the framework can learn nonlinear decision boundaries.

The XOR function is not linearly separable, so a single-layer network cannot solve it.
This example demonstrates that a 2-layer MLP with a non-linear activation (ReLU)
can successfully learn the XOR mapping.

Truth table:
  [0, 0] → 0
  [0, 1] → 1
  [1, 0] → 1
  [1, 1] → 0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from micrograd import Tensor
from micrograd.nn import Sequential, Linear, ReLU, Sigmoid
from micrograd.nn.loss import BCELoss
from micrograd.optim import Adam


def main():
    np.random.seed(42)

    # XOR dataset — all 4 input combinations
    X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], requires_grad=False)
    y = Tensor([[0], [1], [1], [0]], requires_grad=False)

    # 2-layer MLP: 2 → 8 (ReLU) → 1 (Sigmoid)
    model = Sequential(
        Linear(2, 8),
        ReLU(),
        Linear(8, 1),
        Sigmoid()
    )

    optimizer = Adam(list(model.parameters()), lr=0.01)
    criterion = BCELoss()

    print("Training XOR network...")
    print("-" * 40)

    for epoch in range(2000):
        # Forward pass
        pred = model(X)
        loss = criterion(pred, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch + 1:4d}/2000 — Loss: {loss.item():.6f}")

    print("\nPredictions (after training):")
    print("-" * 40)
    model.eval()
    preds = model(X)
    all_correct = True
    for inp, pred_val, target in zip(X.data, preds.data, y.data):
        predicted_class = int(pred_val[0] > 0.5)
        target_class = int(target[0])
        correct = predicted_class == target_class
        if not correct:
            all_correct = False
        status = "OK" if correct else "WRONG"
        print(f"  Input: {inp.astype(int)} -> Raw: {pred_val[0]:.4f} "
              f"-> Pred: {predicted_class} (target={target_class}) [{status}]")

    print()
    if all_correct:
        print("All XOR predictions correct! Network successfully learned XOR.")
    else:
        print("Some predictions are wrong — try training longer or with different seed.")


if __name__ == "__main__":
    main()
