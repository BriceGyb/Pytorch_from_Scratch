"""
Functional operations for Tensors with automatic differentiation support.
All operations return new Tensors and register their backward functions.
"""

import numpy as np
from .tensor import Tensor


def exp(tensor):
    """Element-wise natural exponential: e^x"""
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    exp_val = np.exp(tensor.data)
    out = Tensor(exp_val,
                 requires_grad=tensor.requires_grad,
                 _children=(tensor,), _op='exp')

    def _backward():
        if tensor.requires_grad:
            if tensor.grad is None:
                tensor.grad = np.zeros_like(tensor.data)
            # d/dx exp(x) = exp(x)
            tensor.grad += out.grad * exp_val

    out._backward = _backward
    return out


def log(tensor):
    """Element-wise natural logarithm: ln(x)"""
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    # Clip for numerical stability to avoid log(0) = -inf
    clipped = np.clip(tensor.data, 1e-10, None)
    out = Tensor(np.log(clipped),
                 requires_grad=tensor.requires_grad,
                 _children=(tensor,), _op='log')

    def _backward():
        if tensor.requires_grad:
            if tensor.grad is None:
                tensor.grad = np.zeros_like(tensor.data)
            # d/dx ln(x) = 1/x
            tensor.grad += out.grad / clipped

    out._backward = _backward
    return out


def sqrt(tensor):
    """Element-wise square root: sqrt(x) = x^0.5"""
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    return tensor ** 0.5


def relu(tensor):
    """Rectified Linear Unit: max(0, x)"""
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    out = Tensor(np.maximum(0, tensor.data),
                 requires_grad=tensor.requires_grad,
                 _children=(tensor,), _op='relu')

    def _backward():
        if tensor.requires_grad:
            if tensor.grad is None:
                tensor.grad = np.zeros_like(tensor.data)
            # Gradient passes through where input > 0
            tensor.grad += out.grad * (tensor.data > 0).astype(np.float64)

    out._backward = _backward
    return out


def sigmoid(tensor):
    """Sigmoid activation: 1 / (1 + exp(-x))"""
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    # Numerically stable sigmoid
    s = np.where(tensor.data >= 0,
                 1.0 / (1.0 + np.exp(-tensor.data)),
                 np.exp(tensor.data) / (1.0 + np.exp(tensor.data)))
    out = Tensor(s,
                 requires_grad=tensor.requires_grad,
                 _children=(tensor,), _op='sigmoid')

    def _backward():
        if tensor.requires_grad:
            if tensor.grad is None:
                tensor.grad = np.zeros_like(tensor.data)
            # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
            tensor.grad += out.grad * s * (1.0 - s)

    out._backward = _backward
    return out


def tanh(tensor):
    """Hyperbolic tangent activation: tanh(x)"""
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    t = np.tanh(tensor.data)
    out = Tensor(t,
                 requires_grad=tensor.requires_grad,
                 _children=(tensor,), _op='tanh')

    def _backward():
        if tensor.requires_grad:
            if tensor.grad is None:
                tensor.grad = np.zeros_like(tensor.data)
            # d/dx tanh(x) = 1 - tanh(x)^2
            tensor.grad += out.grad * (1.0 - t ** 2)

    out._backward = _backward
    return out


def softmax(tensor, axis=-1):
    """
    Softmax activation with numerical stability (subtract max before exp).
    Forward: softmax(x)_i = exp(x_i) / sum(exp(x_j))
    Backward: Jacobian-vector product using the identity:
        ds/dx = s * (I - s^T) -> dL/dx = s * (dL/ds - sum(dL/ds * s, axis))
    """
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    # Numerically stable: subtract max along axis
    shifted = tensor.data - tensor.data.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    s = e / e.sum(axis=axis, keepdims=True)
    out = Tensor(s,
                 requires_grad=tensor.requires_grad,
                 _children=(tensor,), _op='softmax')

    def _backward():
        if tensor.requires_grad:
            if tensor.grad is None:
                tensor.grad = np.zeros_like(tensor.data)
            # Jacobian-vector product: dL/dx = s * (dL/ds - sum(dL/ds * s, axis))
            dot = (out.grad * s).sum(axis=axis, keepdims=True)
            tensor.grad += s * (out.grad - dot)

    out._backward = _backward
    return out


def dropout(tensor, p=0.5, training=True):
    """
    Dropout regularization.
    During training: randomly zeros elements with probability p, scales remaining by 1/(1-p).
    During evaluation: returns tensor unchanged.
    """
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    if not training or p == 0.0:
        return tensor

    # Inverted dropout: scale by 1/(1-p) to maintain expected value
    mask = (np.random.rand(*tensor.data.shape) > p).astype(np.float64) / (1.0 - p)
    out = Tensor(tensor.data * mask,
                 requires_grad=tensor.requires_grad,
                 _children=(tensor,), _op='dropout')

    def _backward():
        if tensor.requires_grad:
            if tensor.grad is None:
                tensor.grad = np.zeros_like(tensor.data)
            tensor.grad += out.grad * mask

    out._backward = _backward
    return out


def concat(tensors, axis=0):
    """
    Concatenate tensors along a given axis.
    Backward splits the gradient back to each input tensor.
    """
    tensors = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(np.concatenate([t.data for t in tensors], axis=axis),
                 requires_grad=requires_grad,
                 _children=tuple(tensors), _op='concat')

    # Store the sizes along the concat axis for splitting
    sizes = [t.data.shape[axis] for t in tensors]

    def _backward():
        # Split the gradient back to each input
        split_grads = np.split(out.grad, np.cumsum(sizes)[:-1], axis=axis)
        for t, g in zip(tensors, split_grads):
            if t.requires_grad:
                if t.grad is None:
                    t.grad = np.zeros_like(t.data)
                t.grad += g

    out._backward = _backward
    return out


def stack(tensors, axis=0):
    """
    Stack tensors along a new axis.
    Backward splits and squeezes the gradient back to each input tensor.
    """
    tensors = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(np.stack([t.data for t in tensors], axis=axis),
                 requires_grad=requires_grad,
                 _children=tuple(tensors), _op='stack')

    n = len(tensors)

    def _backward():
        # Split along the stacked axis, then squeeze that axis
        split_grads = np.split(out.grad, n, axis=axis)
        for t, g in zip(tensors, split_grads):
            if t.requires_grad:
                if t.grad is None:
                    t.grad = np.zeros_like(t.data)
                t.grad += np.squeeze(g, axis=axis)

    out._backward = _backward
    return out
