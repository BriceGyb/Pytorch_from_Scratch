"""
Tests for the Tensor class and its basic arithmetic operations.
Gradient correctness is verified by comparing against numerical (finite difference) gradients.
"""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from micrograd import Tensor


def numerical_gradient(f, x, eps=1e-5):
    """
    Compute numerical gradient of f w.r.t. x using centered finite differences:
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


class TestTensorBasicOps:
    """Test basic arithmetic operations and their gradients."""

    def test_add_backward(self):
        """d/dx (x + y).sum() = 1 for all x."""
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        y = Tensor(np.random.randn(3, 4), requires_grad=True)
        out = (x + y).sum()
        out.backward()
        num_grad = numerical_gradient(lambda t: (t + y).sum(), x)
        assert np.allclose(x.grad, num_grad, atol=1e-5)

    def test_mul_backward(self):
        """d/dx (x * y).sum() = y."""
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        y = Tensor(np.random.randn(3, 4), requires_grad=True)
        out = (x * y).sum()
        out.backward()
        num_grad = numerical_gradient(lambda t: (t * y).sum(), x)
        assert np.allclose(x.grad, num_grad, atol=1e-5)

    def test_matmul_backward(self):
        """Gradient of matrix multiplication."""
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        w = Tensor(np.random.randn(4, 5), requires_grad=True)
        out = (x @ w).sum()
        out.backward()
        num_grad = numerical_gradient(lambda t: (t @ w).sum(), x)
        assert np.allclose(x.grad, num_grad, atol=1e-5)

    def test_matmul_weight_backward(self):
        """Gradient of matrix multiplication w.r.t. weight."""
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        w = Tensor(np.random.randn(4, 5), requires_grad=True)
        out = (x @ w).sum()
        out.backward()
        num_grad = numerical_gradient(lambda t: (x @ t).sum(), w)
        assert np.allclose(w.grad, num_grad, atol=1e-5)

    def test_pow_backward(self):
        """d/dx x^2 = 2x."""
        x = Tensor(np.random.randn(3, 4) + 2.0, requires_grad=True)
        out = (x ** 2).sum()
        out.backward()
        num_grad = numerical_gradient(lambda t: (t ** 2).sum(), x)
        assert np.allclose(x.grad, num_grad, atol=1e-5)

    def test_pow_negative_exponent(self):
        """d/dx x^(-1) = -1/x^2."""
        x = Tensor(np.abs(np.random.randn(3, 4)) + 1.0, requires_grad=True)
        out = (x ** -1).sum()
        out.backward()
        num_grad = numerical_gradient(lambda t: (t ** -1).sum(), x)
        assert np.allclose(x.grad, num_grad, atol=1e-5)

    def test_sub_backward(self):
        """d/dx (x - y).sum() = 1."""
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        y = Tensor(np.random.randn(3, 4), requires_grad=True)
        out = (x - y).sum()
        out.backward()
        num_grad = numerical_gradient(lambda t: (t - y).sum(), x)
        assert np.allclose(x.grad, num_grad, atol=1e-5)

    def test_neg_backward(self):
        """d/dx (-x).sum() = -1."""
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        out = (-x).sum()
        out.backward()
        num_grad = numerical_gradient(lambda t: (-t).sum(), x)
        assert np.allclose(x.grad, num_grad, atol=1e-5)

    def test_div_backward(self):
        """d/dx (x / c) = 1/c."""
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        out = (x / 2.0).sum()
        out.backward()
        num_grad = numerical_gradient(lambda t: (t / 2.0).sum(), x)
        assert np.allclose(x.grad, num_grad, atol=1e-5)

    def test_broadcasting_add(self):
        """Test gradient shapes are preserved after broadcasting."""
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(4), requires_grad=True)
        out = (x + b).sum()
        out.backward()
        assert x.grad.shape == x.shape
        assert b.grad.shape == b.shape

    def test_broadcasting_mul(self):
        """Test broadcasting in multiplication."""
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        s = Tensor(np.array(2.0), requires_grad=True)
        out = (x * s).sum()
        out.backward()
        assert x.grad.shape == x.shape
        assert s.grad.shape == s.shape

    def test_mean_backward(self):
        """d/dx mean(x) = 1/n for all x."""
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        out = x.mean()
        out.backward()
        num_grad = numerical_gradient(lambda t: t.mean(), x)
        assert np.allclose(x.grad, num_grad, atol=1e-5)

    def test_mean_axis_backward(self):
        """Test mean with axis argument."""
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        out = x.mean(axis=0).sum()
        out.backward()
        num_grad = numerical_gradient(lambda t: t.mean(axis=0).sum(), x)
        assert np.allclose(x.grad, num_grad, atol=1e-5)

    def test_sum_backward(self):
        """d/dx sum(x) = 1 for all x."""
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        out = x.sum()
        out.backward()
        assert np.allclose(x.grad, np.ones((3, 4)), atol=1e-10)

    def test_sum_axis_backward(self):
        """Test sum with axis argument."""
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        out = x.sum(axis=1).sum()
        out.backward()
        num_grad = numerical_gradient(lambda t: t.sum(axis=1).sum(), x)
        assert np.allclose(x.grad, num_grad, atol=1e-5)

    def test_reshape_backward(self):
        """Test gradient flows correctly through reshape."""
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        out = x.reshape(12).sum()
        out.backward()
        assert x.grad.shape == (3, 4)
        assert np.allclose(x.grad, np.ones((3, 4)))

    def test_chain_of_ops(self):
        """Test gradient through a chain of operations."""
        x = Tensor(np.random.randn(4), requires_grad=True)
        # f(x) = sum(x^2 + 2*x + 1)
        out = (x ** 2 + x * 2 + 1).sum()
        out.backward()
        num_grad = numerical_gradient(lambda t: (t ** 2 + t * 2 + 1).sum(), x)
        assert np.allclose(x.grad, num_grad, atol=1e-5)

    def test_zero_grad(self):
        """Test that zero_grad resets gradients to zero."""
        x = Tensor(np.random.randn(3), requires_grad=True)
        out = (x ** 2).sum()
        out.backward()
        assert x.grad is not None
        x.zero_grad()
        assert np.allclose(x.grad, np.zeros(3))

    def test_detach(self):
        """Test detach creates a new Tensor without grad tracking."""
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        y = x.detach()
        assert not y.requires_grad
        assert np.allclose(y.data, x.data)

    def test_item(self):
        """Test item() returns a Python scalar."""
        x = Tensor(np.array(3.14))
        assert isinstance(x.item(), float)
        assert abs(x.item() - 3.14) < 1e-10

    def test_numpy(self):
        """Test numpy() returns the underlying array."""
        data = np.array([1.0, 2.0, 3.0])
        x = Tensor(data)
        arr = x.numpy()
        assert isinstance(arr, np.ndarray)
        assert np.allclose(arr, data)

    def test_transpose_property(self):
        """Test .T property transposes correctly."""
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        out = x.T.sum()
        out.backward()
        assert x.grad.shape == (3, 4)
        assert np.allclose(x.grad, np.ones((3, 4)))

    def test_gradient_accumulation(self):
        """Test that gradients accumulate correctly across multiple backward calls."""
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        x.zero_grad()

        # First backward pass
        out1 = (x * 2).sum()
        out1.backward()
        grad_after_1 = x.grad.copy()

        # Second backward pass (accumulates)
        out2 = (x * 3).sum()
        out2.backward()

        expected = grad_after_1 + np.ones(3) * 3
        assert np.allclose(x.grad, expected)
