"""
Optimizer implementations for neural network training.
All optimizers follow the same interface:
  - __init__(params, lr, ...)
  - step()        — update parameters using current gradients
  - zero_grad()   — reset all parameter gradients to zero
"""

import numpy as np


class SGD:
    """
    Stochastic Gradient Descent with optional momentum and weight decay.

    Update rule (without momentum):
        param = param - lr * grad

    Update rule (with momentum):
        velocity = momentum * velocity - lr * grad
        param = param + velocity

    Update rule (with weight decay):
        grad_effective = grad + weight_decay * param
        param = param - lr * grad_effective

    Parameters
    ----------
    params : list of Tensor
        Tensors with requires_grad=True to optimize.
    lr : float
        Learning rate (step size). Default: 0.01
    momentum : float
        Momentum factor. 0 disables momentum. Default: 0.0
    weight_decay : float
        L2 regularization coefficient. 0 disables. Default: 0.0
    """

    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Initialize velocity buffers for momentum
        if momentum != 0.0:
            self.velocities = [np.zeros_like(p.data) for p in self.params]
        else:
            self.velocities = None

    def step(self):
        """Perform a single optimization step (parameter update)."""
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.copy()

            # Apply weight decay (L2 regularization)
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * param.data

            # Apply momentum
            if self.momentum != 0.0 and self.velocities is not None:
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad
                param.data += self.velocities[i]
            else:
                param.data -= self.lr * grad

    def zero_grad(self):
        """Reset all parameter gradients to zero."""
        for param in self.params:
            param.zero_grad()

    def __repr__(self):
        return (f"SGD(lr={self.lr}, momentum={self.momentum}, "
                f"weight_decay={self.weight_decay})")


class Adam:
    """
    Adaptive Moment Estimation (Adam) optimizer.

    Update rule:
        m_t = beta1 * m_{t-1} + (1 - beta1) * grad         (1st moment)
        v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2       (2nd moment)
        m_hat = m_t / (1 - beta1^t)                         (bias correction)
        v_hat = v_t / (1 - beta2^t)                         (bias correction)
        param = param - lr * m_hat / (sqrt(v_hat) + eps)

    Parameters
    ----------
    params : list of Tensor
        Tensors with requires_grad=True to optimize.
    lr : float
        Learning rate (step size). Default: 0.001
    betas : tuple (float, float)
        Coefficients for computing running averages of gradient and its square.
        Default: (0.9, 0.999)
    eps : float
        Small value for numerical stability. Default: 1e-8
    weight_decay : float
        L2 regularization coefficient. 0 disables. Default: 0.0
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Step counter (for bias correction)
        self.t = 0

        # Initialize first and second moment buffers
        self.m = [np.zeros_like(p.data) for p in self.params]  # 1st moment (mean)
        self.v = [np.zeros_like(p.data) for p in self.params]  # 2nd moment (variance)

    def step(self):
        """Perform a single optimization step (parameter update)."""
        self.t += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.copy()

            # Apply weight decay (L2 regularization)
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * param.data

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad

            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad ** 2)

            # Compute bias-corrected moment estimates
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)

            # Update parameters
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Reset all parameter gradients to zero."""
        for param in self.params:
            param.zero_grad()

    def __repr__(self):
        return (f"Adam(lr={self.lr}, betas=({self.beta1}, {self.beta2}), "
                f"eps={self.eps}, weight_decay={self.weight_decay})")
