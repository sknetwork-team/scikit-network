from __future__ import annotations

"""
Created on Thu Apr 21 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""
from typing import Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sknetwork.gnn.base import BaseGNN


class BaseOptimizer:
    """Base class for optimizers.

    Parameters
    ----------
    learning_rate: float (default = 0.01)
        Learning rate for updating weights.
    """

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, gnn: BaseGNN):
        """Update model parameters according to gradient values.

        Parameters
        ----------
        gnn: BaseGNNClassifier
            Model containing parameters to update.
        """


class GD(BaseOptimizer):
    """Gradient Descent optimizer.

    Parameters
    ----------
    learning_rate: float (default = 0.01)
        Learning rate for updating weights.
    """

    def __init__(self, learning_rate: float = 0.01):
        super(GD, self).__init__(learning_rate)

    def step(self, gnn: BaseGNN):
        """Update model parameters according to gradient values.

        Parameters
        ----------
        gnn: BaseGNNClassifier
            Model containing parameters to update.
        """
        for idx, layer in enumerate(gnn.layers):
            layer.weight = layer.weight - self.learning_rate * gnn.derivative_weight[idx]
            layer.bias = layer.bias - self.learning_rate * gnn.derivative_bias[idx]


class ADAM(BaseOptimizer):
    """Adam optimizer.

    Parameters
    ----------
    learning_rate: float (default = 0.01)
        Learning rate for updating weights.
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

    def __init__(self, learning_rate: float = 0.01, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8):
        super(ADAM, self).__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_derivative_weight, self.v_derivative_weight = [], []
        self.m_derivative_bias, self.v_derivative_bias = [], []
        self.t = 0

    def step(self, gnn: BaseGNN):
        """Update model parameters according to gradient values and parameters.

        Parameters
        ----------
        gnn: `BaseGNNClassifier`
            Model containing parameters to update.
        """
        if self.t == 0:
            self.m_derivative_weight, self.v_derivative_weight = \
                [np.zeros(x.shape) for x in gnn.derivative_weight], [np.zeros(x.shape) for x in gnn.derivative_weight]
            self.m_derivative_bias, self.v_derivative_bias = \
                [np.zeros(x.shape) for x in gnn.derivative_bias], [np.zeros(x.shape) for x in gnn.derivative_bias]

        for idx, layer in enumerate(gnn.layers):
            self.t += 1

            # Moving averages
            self.m_derivative_weight[idx] = \
                self.beta1 * self.m_derivative_weight[idx] + (1 - self.beta1) * gnn.derivative_weight[idx]
            self.m_derivative_bias[idx] = \
                self.beta1 * self.m_derivative_bias[idx] + (1 - self.beta1) * gnn.derivative_bias[idx]

            self.v_derivative_weight[idx] = \
                self.beta2 * self.v_derivative_weight[idx] + (1 - self.beta2) * (gnn.derivative_weight[idx] ** 2)
            self.v_derivative_bias[idx] = \
                self.beta2 * self.v_derivative_bias[idx] + (1 - self.beta2) * (gnn.derivative_bias[idx] ** 2)

            # Correcting moving averages
            denom_1 = (1 - self.beta1 ** self.t)
            denom_2 = (1 - self.beta2 ** self.t)

            m_derivative_weight_corr = self.m_derivative_weight[idx] / denom_1
            m_derivative_bias_corr = self.m_derivative_bias[idx] / denom_1
            v_derivative_weight_corr = self.v_derivative_weight[idx] / denom_2
            v_derivative_bias_corr = self.v_derivative_bias[idx] / denom_2

            # Parameters update
            layer.weight = \
                layer.weight - (self.learning_rate * m_derivative_weight_corr) / (np.sqrt(v_derivative_weight_corr)
                                                                                  + self.eps)
            if layer.use_bias:
                layer.bias = \
                    layer.bias - (self.learning_rate * m_derivative_bias_corr) / (np.sqrt(v_derivative_bias_corr)
                                                                                  + self.eps)


def get_optimizer(optimizer: Union[BaseOptimizer, str] = 'Adam', learning_rate: float = 0.01) -> BaseOptimizer:
    """Instantiate optimizer according to parameters.

    Parameters
    ----------
    optimizer : str or optimizer
        Which optimizer to use. Can be ``'Adam'`` or ``'GD'`` or custom optimizer.
    learning_rate: float
        Learning rate.

    Returns
    -------
    Optimizer object
    """
    if issubclass(type(optimizer), BaseOptimizer):
        return optimizer
    elif type(optimizer) == str:
        optimizer = optimizer.lower()
        if optimizer == 'adam':
            return ADAM(learning_rate=learning_rate)
        elif optimizer in ['gd', 'gradient']:
            return GD(learning_rate=learning_rate)
        else:
            raise ValueError("Optimizer must be either \"Adam\" or \"GD\" (Gradient Descent).")
    else:
        raise TypeError("Optimizer must be either an \"BaseOptimizer\" object or a string.")
