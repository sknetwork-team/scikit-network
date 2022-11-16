#!/usr/bin/env python3
# coding: utf-8
"""
Created in April 2022
@author: Simon Delarue <sdelarue@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""

from typing import Union

import numpy as np
from scipy import special

from sknetwork.gnn.base_activation import BaseActivation


class ReLu(BaseActivation):
    """ReLu (Rectified Linear Unit) activation function:

    :math:`\\sigma(x) = \\max(0, x)`
    """
    def __init__(self):
        super(ReLu, self).__init__('ReLu')

    @staticmethod
    def output(signal: np.ndarray) -> np.ndarray:
        """Output of the ReLu function."""
        return np.maximum(signal, 0)

    @staticmethod
    def gradient(signal: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Gradient of the ReLu function."""
        return direction * (signal > 0)


class Sigmoid(BaseActivation):
    """Sigmoid activation function:

    :math:`\\sigma(x) = \\frac{1}{1+e^{-x}}`
    Also known as the logistic function.
    """
    def __init__(self):
        super(Sigmoid, self).__init__('Sigmoid')

    @staticmethod
    def output(signal: np.ndarray) -> np.ndarray:
        """Output of the sigmoid function."""
        return special.expit(signal)

    @staticmethod
    def gradient(signal: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Gradient of the sigmoid function."""
        output = Sigmoid.output(signal)
        return output * (1 - output) * direction


class Softmax(BaseActivation):
    """Softmax activation function:

    :math:`\\sigma(x) =
    (\\frac{e^{x_1}}{\\sum_{i=1}^N e^{x_i})},\\ldots,\\frac{e^{x_N}}{\\sum_{i=1}^N e^{x_i})})`

    where :math:`N` is the number of channels.
    """
    def __init__(self):
        super(Softmax, self).__init__('Softmax')

    @staticmethod
    def output(signal: np.ndarray) -> np.ndarray:
        """Output of the softmax function (rows sum to 1)."""
        return special.softmax(signal, axis=1)

    @staticmethod
    def gradient(signal: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Gradient of the softmax function."""
        output = Softmax.output(signal)
        return output * (direction.T - (output * direction).sum(axis=1)).T


def get_activation(activation: Union[BaseActivation, str] = 'identity') -> BaseActivation:
    """Get the activation function.

    Parameters
    ----------
    activation : Union[BaseActivation, str]
        Activation function.
        If a name is given, can be either ``'Identity'``, ``'Relu'``, ``'Sigmoid'`` or ``'Softmax'``.
        If a custom activation function is given, must be of class BaseActivation.

    Returns
    -------
    activation : BaseActivation
        Activation function.

    Raises
    ------
    TypeError
        Error raised if the input not a string or an object of class BaseActivation.
    ValueError
        Error raised if the name of the activation function is unknown.
    """
    if issubclass(type(activation), BaseActivation):
        return activation
    elif type(activation) == str:
        activation = activation.lower()
        if activation in ['identity', '']:
            return BaseActivation()
        elif activation == 'relu':
            return ReLu()
        elif activation == 'sigmoid':
            return Sigmoid()
        elif activation == 'softmax':
            return Softmax()
        else:
            raise ValueError("Activation must be either \"Identity\", \"ReLu\", \"Sigmoid\" or \"Softmax\".")
    else:
        raise TypeError("Activation must be a string or an object of type \"BaseActivation\".")
