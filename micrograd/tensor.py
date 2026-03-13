import numpy as np


def _unbroadcast(grad, original_shape):
    """
    Sum gradient along axes that were broadcast to produce the current shape.
    This is needed when a smaller tensor was broadcast to match a larger one.
    """
    # If shapes are already the same, no unbroadcast needed
    if grad.shape == original_shape:
        return grad

    # Handle scalar original shape
    if original_shape == () or original_shape == (1,):
        return grad.sum().reshape(original_shape) if original_shape == () else grad.sum(keepdims=True).reshape(original_shape)

    # Pad original shape on the left with 1s to match grad ndim
    ndim_diff = grad.ndim - len(original_shape)
    padded_shape = (1,) * ndim_diff + tuple(original_shape)

    # Sum over axes where original had size 1 (were broadcast)
    axes_to_sum = []
    for i, (g_dim, o_dim) in enumerate(zip(grad.shape, padded_shape)):
        if o_dim == 1 and g_dim != 1:
            axes_to_sum.append(i)

    if axes_to_sum:
        grad = grad.sum(axis=tuple(axes_to_sum), keepdims=True)

    # Sum over leading dimensions that were added
    if ndim_diff > 0:
        grad = grad.sum(axis=tuple(range(ndim_diff)))

    # Reshape to original shape
    grad = grad.reshape(original_shape)
    return grad


class Tensor:
    """
    A multi-dimensional array with automatic differentiation support.
    Mimics a subset of PyTorch's Tensor API using NumPy as the backend.
    """

    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        """Transpose for 2-D tensors (or general via numpy)."""
        out = Tensor(self.data.T, requires_grad=self.requires_grad,
                     _children=(self,), _op='T')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad.T

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Arithmetic operations
    # ------------------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='+')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='*')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='@')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __pow__(self, exp):
        assert isinstance(exp, (int, float)), "Only scalar exponents supported"
        out = Tensor(self.data ** exp,
                     requires_grad=self.requires_grad,
                     _children=(self,), _op=f'**{exp}')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad * exp * (self.data ** (exp - 1))

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other if isinstance(other, Tensor) else Tensor(-other))

    def __truediv__(self, other):
        return self * (other ** -1 if isinstance(other, Tensor) else Tensor(other) ** -1)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Tensor(other) - self

    def __rtruediv__(self, other):
        return Tensor(other) / self

    # ------------------------------------------------------------------
    # Reduction operations
    # ------------------------------------------------------------------

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op='sum')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                if axis is None:
                    # Scalar output — broadcast grad back
                    self.grad += np.ones_like(self.data) * out.grad
                else:
                    # Need to expand reduced dims back
                    g = out.grad
                    if not keepdims:
                        g = np.expand_dims(g, axis=axis)
                    self.grad += np.broadcast_to(g, self.data.shape).copy()

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op='mean')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                if axis is None:
                    n = self.data.size
                    self.grad += np.ones_like(self.data) * out.grad / n
                else:
                    n = self.data.shape[axis]
                    g = out.grad
                    if not keepdims:
                        g = np.expand_dims(g, axis=axis)
                    self.grad += np.broadcast_to(g, self.data.shape).copy() / n

        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        out_data = self.data.max(axis=axis, keepdims=keepdims)
        out = Tensor(out_data,
                     requires_grad=self.requires_grad,
                     _children=(self,), _op='max')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                if axis is None:
                    # One-hot mask: 1 where data == max
                    mask = (self.data == out.data).astype(np.float64)
                    # Normalize in case of ties
                    mask /= mask.sum()
                    self.grad += mask * out.grad
                else:
                    g = out.grad
                    max_vals = out.data
                    if not keepdims:
                        g = np.expand_dims(g, axis=axis)
                        max_vals = np.expand_dims(max_vals, axis=axis)
                    mask = (self.data == np.broadcast_to(max_vals, self.data.shape)).astype(np.float64)
                    # Normalize ties along the reduction axis
                    mask /= mask.sum(axis=axis, keepdims=True)
                    self.grad += mask * np.broadcast_to(g, self.data.shape)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Shape operations
    # ------------------------------------------------------------------

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        original_shape = self.data.shape
        out = Tensor(self.data.reshape(shape),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op='reshape')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad.reshape(original_shape)

        out._backward = _backward
        return out

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        out = Tensor(self.data.transpose(axes),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op='transpose')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                if axes is None:
                    self.grad += out.grad.transpose()
                else:
                    # Inverse permutation
                    inv_axes = np.argsort(axes)
                    self.grad += out.grad.transpose(inv_axes)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self):
        """
        Compute gradients via reverse-mode automatic differentiation.
        Uses topological sort (DFS) to process nodes in correct order.
        """
        # Initialize gradient of the output (self) to ones if scalar
        if self.data.ndim == 0 or self.data.size == 1:
            if self.grad is None:
                self.grad = np.ones_like(self.data)
        else:
            if self.grad is None:
                self.grad = np.ones_like(self.data)

        # Build topological ordering via DFS
        topo = []
        visited = set()

        def build_topo(node):
            if id(node) not in visited:
                visited.add(id(node))
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        # Reverse pass: call _backward in reverse topological order
        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def zero_grad(self):
        """Reset gradient to zeros."""
        self.grad = np.zeros_like(self.data)

    def item(self):
        """Return the value as a Python scalar."""
        return float(self.data.flat[0])

    def numpy(self):
        """Return the underlying NumPy array."""
        return self.data

    def detach(self):
        """Return a new Tensor with the same data but no gradient tracking."""
        return Tensor(self.data.copy())

    def __repr__(self):
        return (f"Tensor({self.data}, requires_grad={self.requires_grad}"
                + (f", grad_fn=<{self._op}>" if self._op else "") + ")")

    # ------------------------------------------------------------------
    # Additional dunder methods for convenience
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out = Tensor(self.data[idx],
                     requires_grad=self.requires_grad,
                     _children=(self,), _op='index')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad[idx] += out.grad

        out._backward = _backward
        return out
