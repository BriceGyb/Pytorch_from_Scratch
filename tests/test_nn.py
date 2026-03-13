"""
Tests for neural network components: layers, losses, and optimizers.
"""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from micrograd import Tensor
from micrograd.nn import Sequential, Linear, ReLU, Sigmoid, Tanh, Softmax, Dropout
from micrograd.nn.loss import MSELoss, BCELoss, CrossEntropyLoss
from micrograd.optim import SGD, Adam


class TestLinear:
    """Tests for the Linear layer."""

    def test_forward_shape(self):
        """Output shape should be (batch, out_features)."""
        layer = Linear(4, 8)
        x = Tensor(np.random.randn(3, 4))
        out = layer(x)
        assert out.shape == (3, 8)

    def test_no_bias_shape(self):
        """Linear without bias should still produce correct shape."""
        layer = Linear(4, 8, bias=False)
        x = Tensor(np.random.randn(5, 4))
        out = layer(x)
        assert out.shape == (5, 8)

    def test_backward(self):
        """All gradients should be computed after backward pass."""
        layer = Linear(4, 8)
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        out = layer(x).sum()
        out.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        for p in layer.parameters():
            assert p.grad is not None

    def test_parameter_count(self):
        """Linear(in, out) should have in*out + out parameters."""
        layer = Linear(4, 8, bias=True)
        params = list(layer.parameters())
        total = sum(p.data.size for p in params)
        assert total == 4 * 8 + 8  # weight + bias

    def test_no_bias_parameter_count(self):
        """Linear(in, out, bias=False) should have only in*out parameters."""
        layer = Linear(4, 8, bias=False)
        params = list(layer.parameters())
        total = sum(p.data.size for p in params)
        assert total == 4 * 8

    def test_xavier_init_range(self):
        """Xavier init: weights should be in [-limit, limit]."""
        layer = Linear(100, 100)
        limit = np.sqrt(6.0 / (100 + 100))
        assert np.all(layer.weight.data >= -limit - 1e-10)
        assert np.all(layer.weight.data <= limit + 1e-10)

    def test_bias_init_zero(self):
        """Bias should be initialized to zero."""
        layer = Linear(4, 8)
        assert np.allclose(layer.bias.data, np.zeros(8))


class TestActivationLayers:
    """Tests for activation function layers."""

    def test_relu_forward(self):
        x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
        out = ReLU()(x)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        assert np.allclose(out.data, expected)

    def test_sigmoid_range(self):
        x = Tensor(np.random.randn(10))
        out = Sigmoid()(x)
        assert np.all(out.data > 0) and np.all(out.data < 1)

    def test_tanh_range(self):
        x = Tensor(np.random.randn(10))
        out = Tanh()(x)
        assert np.all(out.data > -1) and np.all(out.data < 1)

    def test_softmax_sum(self):
        x = Tensor(np.random.randn(3, 5))
        out = Softmax()(x)
        row_sums = out.data.sum(axis=-1)
        assert np.allclose(row_sums, np.ones(3))


class TestDropout:
    """Tests for Dropout layer."""

    def test_eval_mode_identity(self):
        """In eval mode, dropout should be identity."""
        drop = Dropout(0.5)
        drop.eval()
        x = Tensor(np.ones((100, 100)))
        out = drop(x)
        assert np.allclose(out.data, x.data)

    def test_training_mode_zeros(self):
        """In training mode, some elements should be zeroed."""
        np.random.seed(0)
        drop = Dropout(0.5)
        drop.train()
        x = Tensor(np.ones((1000,)))
        out = drop(x)
        zero_frac = (out.data == 0).mean()
        assert 0.4 < zero_frac < 0.6

    def test_no_parameters(self):
        """Dropout should have no learnable parameters."""
        drop = Dropout(0.5)
        params = list(drop.parameters())
        assert len(params) == 0


class TestSequential:
    """Tests for Sequential container."""

    def test_forward(self):
        """Sequential should apply layers in order."""
        model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
        x = Tensor(np.random.randn(5, 4))
        out = model(x)
        assert out.shape == (5, 2)

    def test_parameter_collection(self):
        """parameters() should yield all params from all sub-layers."""
        model = Sequential(Linear(4, 8), Linear(8, 2))
        params = list(model.parameters())
        # Linear(4,8) has 4*8+8=40, Linear(8,2) has 8*2+2=18 → total 58
        total = sum(p.data.size for p in params)
        assert total == 40 + 18

    def test_train_eval_mode(self):
        """train() and eval() should propagate to all sub-modules."""
        model = Sequential(Linear(4, 8), Dropout(0.5))
        model.eval()
        # All modules should be in eval mode
        modules = model._modules
        for m in modules.values():
            assert not m.training

        model.train()
        for m in modules.values():
            assert m.training

    def test_zero_grad(self):
        """zero_grad should reset all parameter gradients."""
        model = Sequential(Linear(4, 8), Linear(8, 2))
        x = Tensor(np.random.randn(3, 4))
        out = model(x).sum()
        out.backward()

        # Some grads should be non-None after backward
        grads_before = [p.grad for p in model.parameters()]
        assert any(g is not None for g in grads_before)

        model.zero_grad()
        for p in model.parameters():
            assert np.allclose(p.grad, np.zeros_like(p.data))

    def test_xor_convergence(self):
        """A small MLP should converge on the XOR problem."""
        np.random.seed(42)
        X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], requires_grad=False)
        y = Tensor([[0], [1], [1], [0]], requires_grad=False)

        model = Sequential(Linear(2, 8), ReLU(), Linear(8, 1), Sigmoid())
        optimizer = Adam(list(model.parameters()), lr=0.01)
        criterion = BCELoss()

        for _ in range(3000):
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert loss.item() < 0.05, f"XOR loss {loss.item()} did not converge below 0.05"


class TestLosses:
    """Tests for loss functions."""

    def test_mse_perfect_prediction(self):
        """MSE should be 0 for perfect predictions."""
        pred = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        target = Tensor(np.array([1.0, 2.0, 3.0]))
        loss = MSELoss()(pred, target)
        assert abs(loss.item()) < 1e-10

    def test_mse_value(self):
        """MSE of [0,0,0] vs [1,1,1] should be 1.0."""
        pred = Tensor(np.array([0.0, 0.0, 0.0]), requires_grad=True)
        target = Tensor(np.array([1.0, 1.0, 1.0]))
        loss = MSELoss()(pred, target)
        assert abs(loss.item() - 1.0) < 1e-10

    def test_mse_backward(self):
        """MSE backward should produce valid gradients."""
        pred = Tensor(np.random.randn(5), requires_grad=True)
        target = Tensor(np.random.randn(5))
        loss = MSELoss()(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape

    def test_bce_loss_value(self):
        """BCE for near-perfect predictions should be near zero."""
        pred = Tensor(np.array([0.99, 0.01, 0.99, 0.01]), requires_grad=True)
        target = Tensor(np.array([1.0, 0.0, 1.0, 0.0]))
        loss = BCELoss()(pred, target)
        assert loss.item() < 0.1

    def test_bce_backward(self):
        """BCE backward should produce valid gradients."""
        pred = Tensor(np.random.rand(4) * 0.8 + 0.1, requires_grad=True)  # (0.1, 0.9)
        target = Tensor(np.random.randint(0, 2, 4).astype(float))
        loss = BCELoss()(pred, target)
        loss.backward()
        assert pred.grad is not None

    def test_cross_entropy_shape(self):
        """CrossEntropyLoss should produce a scalar."""
        logits = Tensor(np.random.randn(4, 10), requires_grad=True)
        targets = np.array([0, 1, 2, 3])
        loss = CrossEntropyLoss()(logits, targets)
        assert loss.data.ndim == 0 or loss.data.size == 1

    def test_cross_entropy_backward(self):
        """CrossEntropyLoss backward should produce gradients for logits."""
        logits = Tensor(np.random.randn(4, 10), requires_grad=True)
        targets = np.array([0, 1, 2, 3])
        loss = CrossEntropyLoss()(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape

    def test_cross_entropy_numerical_stability(self):
        """CrossEntropyLoss should be stable with large logits."""
        logits = Tensor(np.array([[1000.0, 0.0, 0.0]]), requires_grad=True)
        targets = np.array([0])
        loss = CrossEntropyLoss()(logits, targets)
        assert not np.isnan(loss.item())
        assert not np.isinf(loss.item())

    def test_cross_entropy_correct_prediction(self):
        """High logit for correct class should yield low loss."""
        logits = Tensor(np.array([[10.0, 0.0, 0.0, 0.0]]), requires_grad=True)
        targets = np.array([0])
        loss = CrossEntropyLoss()(logits, targets)
        assert loss.item() < 0.001


class TestOptimizers:
    """Tests for SGD and Adam optimizers."""

    def test_sgd_step(self):
        """SGD step should update parameters."""
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        opt = SGD([x], lr=0.1)
        loss = (x * x).sum()
        loss.backward()
        old_data = x.data.copy()
        opt.step()
        assert not np.allclose(x.data, old_data), "Parameters should change after step"

    def test_sgd_direction(self):
        """SGD should move parameters in the direction of negative gradient."""
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        opt = SGD([x], lr=0.1)
        loss = (x * x).sum()
        loss.backward()
        # grad = 2x = [2, 4, 6]
        old_data = x.data.copy()
        opt.step()
        # Should decrease: new = old - 0.1 * 2*old
        expected = old_data - 0.1 * 2 * old_data
        assert np.allclose(x.data, expected)

    def test_sgd_momentum(self):
        """SGD with momentum should produce different updates than vanilla SGD."""
        x1 = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        x2 = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)

        opt1 = SGD([x1], lr=0.1, momentum=0.0)
        opt2 = SGD([x2], lr=0.1, momentum=0.9)

        for _ in range(3):
            loss1 = (x1 * x1).sum()
            loss1.backward()
            opt1.step()

            loss2 = (x2 * x2).sum()
            loss2.backward()
            opt2.step()

        # With momentum, parameters should move further (or differently)
        assert not np.allclose(x1.data, x2.data)

    def test_sgd_zero_grad(self):
        """zero_grad should reset gradients before each step."""
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        opt = SGD([x], lr=0.1)

        # First step
        loss = (x * x).sum()
        loss.backward()
        opt.step()
        x.zero_grad()

        # Second step — should use fresh gradients
        loss = (x * x).sum()
        loss.backward()
        old_data = x.data.copy()
        opt.step()
        # gradient of x^2 at new x should be 2*new_x
        assert not np.allclose(x.data, old_data)

    def test_adam_step(self):
        """Adam step should update parameters."""
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        opt = Adam([x], lr=0.01)
        loss = (x * x).sum()
        loss.backward()
        old_data = x.data.copy()
        opt.step()
        assert not np.allclose(x.data, old_data), "Parameters should change after Adam step"

    def test_adam_convergence(self):
        """Adam should converge on a simple quadratic minimization."""
        np.random.seed(0)
        x = Tensor(np.array([3.0, -2.0, 1.0]), requires_grad=True)
        opt = Adam([x], lr=0.1)

        for _ in range(500):
            # f(x) = sum(x^2), minimum at x = 0
            loss = (x * x).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

        assert np.allclose(x.data, np.zeros(3), atol=1e-3), \
            f"Adam should converge to 0, got {x.data}"

    def test_adam_step_counter(self):
        """Adam's step counter should increment each call."""
        x = Tensor(np.array([1.0]), requires_grad=True)
        opt = Adam([x], lr=0.01)
        assert opt.t == 0
        for i in range(5):
            loss = (x * x).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            assert opt.t == i + 1

    def test_optimizer_with_model(self):
        """Test optimizer works end-to-end with a Sequential model."""
        model = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
        opt = Adam(list(model.parameters()), lr=0.01)
        x = Tensor(np.random.randn(5, 2))
        target = Tensor(np.random.randn(5, 1))

        # Record initial param values
        initial_params = [p.data.copy() for p in model.parameters()]

        loss = MSELoss()(model(x), target)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Verify parameters changed
        updated_params = [p.data for p in model.parameters()]
        changed = any(not np.allclose(a, b)
                      for a, b in zip(initial_params, updated_params))
        assert changed, "Parameters should change after optimization step"


class TestModuleInterface:
    """Tests for the Module base class."""

    def test_parameter_registration(self):
        """Tensors with requires_grad should be registered as parameters."""
        layer = Linear(4, 8)
        params = object.__getattribute__(layer, '_parameters')
        assert 'weight' in params
        assert 'bias' in params

    def test_module_registration(self):
        """Sub-modules should be registered in _modules."""
        model = Sequential(Linear(4, 8), ReLU())
        modules = object.__getattribute__(model, '_modules')
        assert '0' in modules
        assert '1' in modules

    def test_repr_sequential(self):
        """repr should produce human-readable string."""
        model = Sequential(Linear(2, 4), ReLU())
        r = repr(model)
        assert 'Sequential' in r
        assert 'Linear' in r
        assert 'ReLU' in r
