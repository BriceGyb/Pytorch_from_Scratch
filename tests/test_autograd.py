"""
Tests for automatic differentiation through functional operations (ops.py).
All gradient tests use numerical (finite difference) verification.
"""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from micrograd import Tensor
from micrograd import ops


def numerical_gradient(f, x, eps=1e-5):
    """
    Compute numerical gradient using centered finite differences:
        grad[i] ≈ (f(x + eps*e_i) - f(x - eps*e_i)) / (2 * eps)
    """
    grad = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        orig = x.data[idx]

        x.data[idx] = orig + eps
        f_plus = f(x).item()

        x.data[idx] = orig - eps
        f_minus = f(x).item()

        x.data[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


class TestOpsGradients:
    """Test gradients of all functional operations."""

    def test_relu_backward(self):
        """Gradient of ReLU: 1 where x > 0, else 0."""
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        out = ops.relu(x).sum()
        out.backward()
        num_grad = numerical_gradient(lambda t: ops.relu(t).sum(), x)
        assert np.allclose(x.grad, num_grad, atol=1e-5)

    def test_relu_with_negative_inputs(self):
        """ReLU gradient is 0 for negative inputs."""
        x = Tensor(np.array([-2.0, -1.0, 0.5, 1.0]), requires_grad=True)
        out = ops.relu(x).sum()
        out.backward()
        expected = np.array([0.0, 0.0, 1.0, 1.0])
        assert np.allclose(x.grad, expected)

    def test_sigmoid_backward(self):
        """Gradient of sigmoid: sigmoid(x) * (1 - sigmoid(x))."""
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        out = ops.sigmoid(x).sum()
        out.backward()
        num_grad = numerical_gradient(lambda t: ops.sigmoid(t).sum(), x)
        assert np.allclose(x.grad, num_grad, atol=1e-5)

    def test_sigmoid_numerical_stability(self):
        """Sigmoid should be numerically stable for large inputs."""
        x = Tensor(np.array([500.0, -500.0, 0.0]), requires_grad=True)
        out = ops.sigmoid(x).sum()
        out.backward()
        # Should not produce NaN or Inf
        assert not np.any(np.isnan(x.grad))
        assert not np.any(np.isinf(x.grad))

    def test_exp_backward(self):
        """Gradient of exp(x) is exp(x)."""
        x = Tensor(np.random.randn(3, 4) * 0.5, requires_grad=True)
        out = ops.exp(x).sum()
        out.backward()
        num_grad = numerical_gradient(lambda t: ops.exp(t).sum(), x)
        assert np.allclose(x.grad, num_grad, atol=1e-5)

    def test_log_backward(self):
        """Gradient of log(x) is 1/x."""
        x = Tensor(np.random.rand(3, 4) + 0.5, requires_grad=True)
        out = ops.log(x).sum()
        out.backward()
        num_grad = numerical_gradient(lambda t: ops.log(t).sum(), x)
        assert np.allclose(x.grad, num_grad, atol=1e-5)

    def test_log_clipping(self):
        """log should not produce -inf for zero or near-zero inputs."""
        x = Tensor(np.array([0.0, 1e-15, 1.0]), requires_grad=True)
        out = ops.log(x).sum()
        out.backward()
        assert not np.any(np.isnan(x.grad))
        assert not np.any(np.isinf(x.grad))

    def test_tanh_backward(self):
        """Gradient of tanh(x) is 1 - tanh(x)^2."""
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        out = ops.tanh(x).sum()
        out.backward()
        num_grad = numerical_gradient(lambda t: ops.tanh(t).sum(), x)
        assert np.allclose(x.grad, num_grad, atol=1e-5)

    def test_sqrt_backward(self):
        """Gradient of sqrt(x) = 0.5 / sqrt(x)."""
        x = Tensor(np.random.rand(3, 4) + 0.1, requires_grad=True)
        out = ops.sqrt(x).sum()
        out.backward()
        num_grad = numerical_gradient(lambda t: ops.sqrt(t).sum(), x)
        assert np.allclose(x.grad, num_grad, atol=1e-5)

    def test_softmax_backward(self):
        """Gradient of softmax via Jacobian-vector product."""
        x = Tensor(np.random.randn(4, 5), requires_grad=True)
        out = ops.softmax(x).sum()
        out.backward()
        num_grad = numerical_gradient(lambda t: ops.softmax(t).sum(), x)
        assert np.allclose(x.grad, num_grad, atol=1e-4)

    def test_softmax_output_sums_to_one(self):
        """Each row of softmax output should sum to 1."""
        x = Tensor(np.random.randn(3, 5), requires_grad=False)
        s = ops.softmax(x)
        row_sums = s.data.sum(axis=-1)
        assert np.allclose(row_sums, np.ones(3), atol=1e-10)

    def test_softmax_numerical_stability(self):
        """Softmax should be stable with large inputs."""
        x = Tensor(np.array([[1000.0, 1001.0, 999.0]]), requires_grad=True)
        s = ops.softmax(x)
        assert not np.any(np.isnan(s.data))
        assert not np.any(np.isinf(s.data))
        assert np.allclose(s.data.sum(), 1.0)

    def test_chain_rule(self):
        """Test chain rule through nested operations: sigmoid(relu(x))."""
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        out = ops.sigmoid(ops.relu(x)).sum()
        out.backward()
        num_grad = numerical_gradient(lambda t: ops.sigmoid(ops.relu(t)).sum(), x)
        assert np.allclose(x.grad, num_grad, atol=1e-5)

    def test_complex_chain_rule(self):
        """Test chain rule through a more complex computation graph."""
        x = Tensor(np.random.randn(5), requires_grad=True)
        # f(x) = sum(tanh(exp(x) / (1 + exp(x))))
        out = ops.tanh(ops.exp(x) / (ops.exp(x) + 1)).sum()
        out.backward()
        num_grad = numerical_gradient(
            lambda t: ops.tanh(ops.exp(t) / (ops.exp(t) + 1)).sum(), x
        )
        assert np.allclose(x.grad, num_grad, atol=1e-4)

    def test_concat_backward(self):
        """Test gradient flows correctly through concatenation."""
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(3, 4), requires_grad=True)
        out = ops.concat([a, b], axis=0).sum()
        out.backward()
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape
        assert np.allclose(a.grad, np.ones((3, 4)))
        assert np.allclose(b.grad, np.ones((3, 4)))

    def test_stack_backward(self):
        """Test gradient flows correctly through stack."""
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(3, 4), requires_grad=True)
        out = ops.stack([a, b], axis=0).sum()
        out.backward()
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape
        assert np.allclose(a.grad, np.ones((3, 4)))
        assert np.allclose(b.grad, np.ones((3, 4)))

    def test_dropout_eval_mode(self):
        """Dropout in eval mode should be identity."""
        x = Tensor(np.ones((100, 100)), requires_grad=True)
        out = ops.dropout(x, p=0.5, training=False)
        assert np.allclose(out.data, x.data)

    def test_dropout_training_statistics(self):
        """During training, ~50% of elements should be zeroed with p=0.5."""
        np.random.seed(42)
        x = Tensor(np.ones((1000, 10)), requires_grad=False)
        out = ops.dropout(x, p=0.5, training=True)
        # After inverted dropout, nonzero values are scaled by 1/(1-0.5)=2
        # So either 0 or ~2.0
        zero_fraction = (out.data == 0).mean()
        # Should be close to 0.5 (within a few percent)
        assert 0.4 < zero_fraction < 0.6

    def test_gradient_accumulation_through_ops(self):
        """Test that gradients accumulate properly through op graph."""
        x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        # y = relu(x) + sigmoid(x) — x is used twice
        y = (ops.relu(x) + ops.sigmoid(x)).sum()
        y.backward()
        num_grad = numerical_gradient(
            lambda t: (ops.relu(t) + ops.sigmoid(t)).sum(), x
        )
        assert np.allclose(x.grad, num_grad, atol=1e-5)
