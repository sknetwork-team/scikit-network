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
        lr: float (default=0.01)
            Learning rate for weights update.
    """

    def __init__(self, gnn: BaseGNNClassifier, lr: float = 0.01):
        self.gnn = gnn
        self.lr = lr

    def step(self):
        """Update model parameters according to gradient values."""
        for idx, l in enumerate(self.gnn.layers):
            l.W = l.W - self.lr * self.gnn.dW[idx]
            l.bias = l.bias - self.lr * self.gnn.db[idx]


class ADAM:
    """Adam optimizer.

    Parameters
    ----------
    gnn: `BaseGNNClassifier`
        Model containing parameters to update.
    lr: float (default=0.01)
        Learning rate for weights update.
    beta1, beta2: float
        Coefficients used for computing running averages of gradients.
    epsilon: float (default=1e-8)
        Term added to the denominator to improve stability.

    References
    ----------
    Kingma, D. P., & Ba, J. (2014).
    `Adam: A method for stochastic optimization.
    <https://arxiv.org/pdf/1412.6980.pdf>`_
    3rd International Conference for Learning Representation.
    """

    def __init__(self, gnn: BaseGNNClassifier, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8):
        self.gnn = gnn
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_dw, self.v_dw = [], []
        self.m_db, self.v_db = [], []
        self.t = 0

    def step(self):
        """Update model parameters according to gradient values and parameters."""
        if self.t == 0:
            self.m_dw, self.v_dw = [np.zeros(x.shape) for x in self.gnn.dW], [np.zeros(x.shape) for x in self.gnn.dW]
            self.m_db, self.v_db = [np.zeros(x.shape) for x in self.gnn.db], [np.zeros(x.shape) for x in self.gnn.db]

        for idx, l in enumerate(self.gnn.layers):
            self.t += 1

            # Moving averages
            self.m_dw[idx] = self.beta1 * self.m_dw[idx] + (1 - self.beta1) * self.gnn.dW[idx]
            self.m_db[idx] = self.beta1 * self.m_db[idx] + (1 - self.beta1) * self.gnn.db[idx]

            self.v_dw[idx] = self.beta2 * self.v_dw[idx] + (1 - self.beta2) * (self.gnn.dW[idx] ** 2)
            self.v_db[idx] = self.beta2 * self.v_db[idx] + (1 - self.beta2) * (self.gnn.db[idx] ** 2)

            # Correcting moving averages
            denom_1 = (1 - self.beta1 ** self.t)
            denom_2 = (1 - self.beta2 ** self.t)

            m_dw_corr = self.m_dw[idx] / denom_1
            m_db_corr = self.m_db[idx] / denom_1
            v_dw_corr = self.v_dw[idx] / denom_2
            v_db_corr = self.v_db[idx] / denom_2

            # Parameters update
            l.W = l.W - (self.lr * m_dw_corr) / (np.sqrt(v_dw_corr) + self.epsilon)
            l.bias = l.bias - (self.lr * m_db_corr) / (np.sqrt(v_db_corr) + self.epsilon)


def optimizer_factory(gnn: BaseGNNClassifier, opt: str = 'Adam', **kwargs) -> object:
    """Instantiate optimizer according to parameters.

    Parameters
    ----------
    gnn : BaseGNNClassifier
        Model on which optimizers apply the `step` method.
    opt : str ({'none', 'Adam'}, default='Adam')
        Optimizer name.
    lr : float (default=0.01)
        Learning rate for weights update.

    Returns
    -------
    Optimizer object
    """
    if opt == 'Adam':
        return ADAM(gnn, **kwargs)
    elif opt == 'none':
        return GD(gnn, **kwargs)
    else:
        raise ValueError("Optimizer must be either \"ADAM\" or \"none\".")
