from __future__ import annotations

"""
Created on Thu Apr 21 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sknetwork.gnn import BaseGNNClassifier


class GD:
    """Gradient Descent optimizer.

    Parameters
    ----------
    gnn: BaseGNNClassifier
        Model containing parameters to update.
    learning_rate: float (default = 0.01)
        Learning rate for weights update.
    """

    def __init__(self, gnn: BaseGNNClassifier, learning_rate: float = 0.01):
        self.gnn = gnn
        self.learning_rate = learning_rate

    def step(self):
        """Update model parameters according to gradient values."""
        for idx, layer in enumerate(self.gnn.layers):
            layer.weight = layer.weight - self.learning_rate * self.gnn.prime_weight[idx]
            layer.bias = layer.bias - self.learning_rate * self.gnn.prime_bias[idx]


class ADAM:
    """Adam optimizer.

    Parameters
    ----------
    gnn: `BaseGNNClassifier`
        Model containing parameters to update.
    learning_rate: float (default = 0.01)
        Learning rate for weights update.
    beta1, beta2: float
        Coefficients used for computing running averages of gradients.
    eps: float (default = 1e-8)
        Term added to the denominator to improve stability.

    References
    ----------
    Kingma, D. P., & Ba, J. (2014).
    `Adam: A method for stochastic optimization.
    <https://arxiv.org/pdf/1412.6980.pdf>`_
    3rd International Conference for Learning Representation.
    """

    def __init__(self, gnn: BaseGNNClassifier, learning_rate: float = 0.01, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8):
        self.gnn = gnn
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_prime_weight, self.v_prime_weight = [], []
        self.m_prime_bias, self.v_prime_bias = [], []
        self.t = 0

    def step(self):
        """Update model parameters according to gradient values and parameters."""
        if self.t == 0:
            self.m_prime_weight, self.v_prime_weight = \
                [np.zeros(x.shape) for x in self.gnn.prime_weight], [np.zeros(x.shape) for x in self.gnn.prime_weight]
            self.m_prime_bias, self.v_prime_bias = \
                [np.zeros(x.shape) for x in self.gnn.prime_bias], [np.zeros(x.shape) for x in self.gnn.prime_bias]

        for idx, layer in enumerate(self.gnn.layers):
            self.t += 1

            # Moving averages
            self.m_prime_weight[idx] = \
                self.beta1 * self.m_prime_weight[idx] + (1 - self.beta1) * self.gnn.prime_weight[idx]
            self.m_prime_bias[idx] = \
                self.beta1 * self.m_prime_bias[idx] + (1 - self.beta1) * self.gnn.prime_bias[idx]

            self.v_prime_weight[idx] = \
                self.beta2 * self.v_prime_weight[idx] + (1 - self.beta2) * (self.gnn.prime_weight[idx] ** 2)
            self.v_prime_bias[idx] = \
                self.beta2 * self.v_prime_bias[idx] + (1 - self.beta2) * (self.gnn.prime_bias[idx] ** 2)

            # Correcting moving averages
            denom_1 = (1 - self.beta1 ** self.t)
            denom_2 = (1 - self.beta2 ** self.t)

            m_prime_weight_corr = self.m_prime_weight[idx] / denom_1
            m_prime_bias_corr = self.m_prime_bias[idx] / denom_1
            v_prime_weight_corr = self.v_prime_weight[idx] / denom_2
            v_prime_bias_corr = self.v_prime_bias[idx] / denom_2

            # Parameters update
            layer.weight = \
                layer.weight - (self.learning_rate * m_prime_weight_corr) / (np.sqrt(v_prime_weight_corr) + self.eps)
            layer.bias = \
                layer.bias - (self.learning_rate * m_prime_bias_corr) / (np.sqrt(v_prime_bias_corr) + self.eps)


def get_optimizer(gnn: BaseGNNClassifier, opt: str = 'Adam', **kwargs) -> object:
    """Instantiate optimizer according to parameters.

    Parameters
    ----------
    gnn : BaseGNNClassifier
        Model on which optimizers apply the `step` method.
    opt : str
        Which optimizer to use. Can be ``'Adam'`` or ``'None'``.

    Returns
    -------
    Optimizer object
    """
    opt = opt.lower()
    if opt == 'adam':
        return ADAM(gnn, **kwargs)
    elif opt == 'none':
        return GD(gnn, **kwargs)
    else:
        raise ValueError("Optimizer must be either \"Adam\" or \"None\".")
