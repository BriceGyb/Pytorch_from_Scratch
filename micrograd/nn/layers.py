"""
Neural network layer implementations.
All layers inherit from Module and implement forward().
"""

import numpy as np
from .module import Module
from ..tensor import Tensor
from .. import ops


class Linear(Module):
    """
    Fully-connected (dense) layer: y = x @ W^T + b

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool
        If True (default), adds a learnable bias term.

    Weight initialization: Xavier uniform (Glorot uniform)
    Bias initialization: zeros
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Xavier uniform initialization
        # Gain = 1 (linear), limit = sqrt(6 / (fan_in + fan_out))
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weight = Tensor(
            np.random.uniform(-limit, limit, (out_features, in_features)),
            requires_grad=True
        )

        if bias:
            self.bias = Tensor(
                np.zeros(out_features),
                requires_grad=True
            )
        else:
            # Store as non-parameter attribute so it doesn't get registered
            object.__setattr__(self, '_bias_none', None)

    def forward(self, x):
        """
        Forward pass: x @ weight.T + bias

        Parameters
        ----------
        x : Tensor of shape (batch_size, in_features)

        Returns
        -------
        Tensor of shape (batch_size, out_features)
        """
        out = x @ self.weight.T
        if self.use_bias:
            out = out + self.bias
        return out

    def __repr__(self):
        return (f"Linear(in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.use_bias})")


class ReLU(Module):
    """Rectified Linear Unit activation: max(0, x)"""

    def forward(self, x):
        return ops.relu(x)

    def __repr__(self):
        return "ReLU()"


class Sigmoid(Module):
    """Sigmoid activation: 1 / (1 + exp(-x))"""

    def forward(self, x):
        return ops.sigmoid(x)

    def __repr__(self):
        return "Sigmoid()"


class Tanh(Module):
    """Hyperbolic tangent activation: tanh(x)"""

    def forward(self, x):
        return ops.tanh(x)

    def __repr__(self):
        return "Tanh()"


class Softmax(Module):
    """
    Softmax activation along a given axis.

    Parameters
    ----------
    axis : int
        Axis along which to compute softmax (default: -1, i.e., last axis).
    """

    def __init__(self, axis=-1):
        super().__init__()
        object.__setattr__(self, 'axis', axis)

    def forward(self, x):
        return ops.softmax(x, axis=self.axis)

    def __repr__(self):
        return f"Softmax(axis={self.axis})"


class Dropout(Module):
    """
    Dropout regularization layer.

    During training: randomly zeroes elements with probability p,
    scales remaining elements by 1/(1-p) (inverted dropout).
    During evaluation: passes input unchanged.

    Parameters
    ----------
    p : float
        Probability of an element being zeroed. Default: 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        assert 0.0 <= p < 1.0, "Dropout probability must be in [0, 1)"
        object.__setattr__(self, 'p', p)

    def forward(self, x):
        return ops.dropout(x, p=self.p, training=self.training)

    def __repr__(self):
        return f"Dropout(p={self.p})"


class Sequential(Module):
    """
    A sequential container that applies layers one after another.

    Layers are stored in _modules with string keys "0", "1", "2", ...

    Example
    -------
    model = Sequential(
        Linear(2, 8),
        ReLU(),
        Linear(8, 1),
        Sigmoid()
    )
    output = model(input)
    """

    def __init__(self, *layers):
        super().__init__()
        modules = object.__getattribute__(self, '_modules')
        for i, layer in enumerate(layers):
            modules[str(i)] = layer

    def forward(self, x):
        """Pass input through each layer in order."""
        modules = object.__getattribute__(self, '_modules')
        for layer in modules.values():
            x = layer(x)
        return x

    def __repr__(self):
        modules = object.__getattribute__(self, '_modules')
        if not modules:
            return "Sequential()"
        lines = ["Sequential("]
        for key, module in modules.items():
            mod_str = repr(module).replace('\n', '\n  ')
            lines.append(f"  ({key}): {mod_str}")
        lines.append(")")
        return '\n'.join(lines)
