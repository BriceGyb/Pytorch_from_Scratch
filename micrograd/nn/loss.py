"""
Loss function implementations for neural network training.
All losses are callable Modules that compute scalar loss values.
"""

import numpy as np
from .module import Module
from ..tensor import Tensor
from .. import ops


class MSELoss(Module):
    """
    Mean Squared Error Loss: L = mean((pred - target)^2)

    Used for regression tasks.

    Parameters
    ----------
    pred : Tensor
        Model predictions.
    target : Tensor or array-like
        Ground truth values, same shape as pred.

    Returns
    -------
    Tensor : scalar loss value
    """

    def forward(self, pred, target):
        if not isinstance(target, Tensor):
            target = Tensor(target)
        diff = pred - target
        return (diff ** 2).mean()

    def __repr__(self):
        return "MSELoss()"


class BCELoss(Module):
    """
    Binary Cross-Entropy Loss:
        L = -mean(target * log(pred) + (1 - target) * log(1 - pred))

    Used for binary classification tasks.
    Inputs should be probabilities in (0, 1), e.g., after Sigmoid.

    Parameters
    ----------
    pred : Tensor
        Predicted probabilities (values in [0, 1]).
    target : Tensor or array-like
        Binary targets (0 or 1).

    Returns
    -------
    Tensor : scalar loss value
    """

    def forward(self, pred, target):
        if not isinstance(target, Tensor):
            target = Tensor(target)
        # Add small epsilon for numerical stability to avoid log(0)
        eps = 1e-8
        loss = -(target * ops.log(pred + eps) + (1 - target) * ops.log(1 - pred + eps))
        return loss.mean()

    def __repr__(self):
        return "BCELoss()"


class CrossEntropyLoss(Module):
    """
    Cross-Entropy Loss for multi-class classification.
    Combines log-softmax and negative log-likelihood in a numerically stable way.

        L = -mean(log_softmax(logits)[range(N), targets])

    where log_softmax is computed as:
        log_softmax(x) = x - log(sum(exp(x - max(x)))) - max(x)
                       = x - max(x) - log(sum(exp(x - max(x))))

    Parameters
    ----------
    logits : Tensor of shape (N, C)
        Raw, unnormalized scores for each class (before softmax).
    targets : array-like of shape (N,)
        Integer class indices in range [0, C).

    Returns
    -------
    Tensor : scalar loss value
    """

    def forward(self, logits, targets):
        """
        Numerically stable cross-entropy using the log-sum-exp trick.
        Implemented directly in terms of Tensor operations for autograd.
        """
        if not isinstance(logits, Tensor):
            logits = Tensor(logits)

        targets = np.array(targets, dtype=np.int64)
        N, C = logits.data.shape

        # Numerically stable log-softmax:
        # log_softmax(x)_i = x_i - max(x) - log(sum(exp(x_j - max(x))))
        # We implement this using our Tensor ops to get correct gradients.

        # Step 1: subtract max for numerical stability (no gradient through this)
        max_logits = logits.data.max(axis=1, keepdims=True)  # (N, 1), no grad

        # Step 2: shifted = logits - max_logits  (Tensor op)
        shifted = logits - Tensor(max_logits)  # (N, C)

        # Step 3: log(sum(exp(shifted))) per sample  (Tensor op)
        log_sum_exp = ops.log(ops.exp(shifted).sum(axis=1, keepdims=True))  # (N, 1)

        # Step 4: log_softmax = shifted - log_sum_exp  (Tensor op)
        log_probs = shifted - log_sum_exp  # (N, C)

        # Step 5: gather the log-probabilities for the true classes
        # log_probs.data has shape (N, C); we pick log_probs[i, targets[i]]
        correct_log_probs = log_probs.data[np.arange(N), targets]  # (N,) numpy array

        # Wrap as Tensor and set up backward through log_probs
        # We need the NLL loss to be differentiable w.r.t. log_probs
        # Loss = -mean(correct_log_probs)
        # dL/d(log_probs[i,j]) = -1/N if j == targets[i] else 0

        nll_data = -correct_log_probs.mean()  # Python scalar
        out = Tensor(nll_data,
                     requires_grad=logits.requires_grad,
                     _children=(log_probs,), _op='cross_entropy')

        def _backward():
            if log_probs.requires_grad:
                if log_probs.grad is None:
                    log_probs.grad = np.zeros_like(log_probs.data)
                # Gradient of NLL w.r.t. log_probs
                grad = np.zeros((N, C), dtype=np.float64)
                grad[np.arange(N), targets] = -1.0 / N
                log_probs.grad += grad * out.grad

        out._backward = _backward
        return out

    def __repr__(self):
        return "CrossEntropyLoss()"
