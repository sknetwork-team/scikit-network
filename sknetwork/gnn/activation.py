#!/usr/bin/env python3
# coding: utf-8
"""
Created on Thu Apr 21 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""

from typing import Callable

import numpy as np
from scipy import special


def relu(input: np.ndarray) -> np.ndarray:
    """Apply the Rectified Linear Unit function pointwise:

    :math:`\text{ReLU}(x) = (x)^+ = max(0, x)`
    """

    return np.maximum(input, 0)


def sigmoid(input: np.ndarray) -> np.ndarray:
    """Apply the logistic function pointwise:

    :math:`\text{sigmoid}(x) = \dfrac{1}{(1+e^{-x})}`

    Note: We use the `expit` function from `scipy.special`.
    """

    return special.expit(input)


def softmax(input: np.ndarray) -> np.ndarray:
    """Apply the softmax function on each row.

    Note: We use `softmax` function from `scipy.special`.

    Returns
    -------
    np.ndarray
        Output array with same shape as `input`. Rows will sum to 1.
    """

    return special.softmax(input, axis=1)


def get_activation_function(activation_name: str) -> Callable[..., float]:
    """Returns activation function according to `activation_name`.

    Parameters
    ----------
    activation_name : str
        Which activation function to use. Can be ``'Relu'``, ``'Sigmoid'`` or ``'Softmax'``.

    Returns
    -------
    Callable[..., float]
        Activation function

    Raises
    ------
    ValueError
        Error raised if activation function does not exist.
    """
    activation_name = activation_name.lower()
    if activation_name == 'relu':
        return relu
    elif activation_name == 'sigmoid':
        return sigmoid
    elif activation_name == 'softmax':
        return softmax
    else:
        raise ValueError("Activation must be either \"Relu\", \"Sigmoid\" or \"Softmax\"")


def relu_prime(input: np.ndarray) -> np.ndarray:
    """Derivative of the Rectified Linear Unit function."""
    return 1 * (input > 0)


def sigmoid_prime(input: np.ndarray) -> np.ndarray:
    """Derivative of the sigmoid (a.k.a. logistic) function."""
    return sigmoid(input) * (1 - sigmoid(input))


def get_prime_activation_function(activation_name: str) -> Callable[..., float]:
    """Returns activation function derivative according to `activation_name`.

    Parameters
    ----------
    activation_name : str
        Which activation function derivative to use. Can be ``'Relu'``, ``'Sigmoid'`` or ``'Softmax'``.

    Returns
    -------
    Callable[..., float]
        Activation function

    Raises
    ------
    ValueError
        Error raised if activation function does not exist.
    """
    activation_name = activation_name.lower()
    if activation_name == 'relu':
        return relu_prime
    elif activation_name == 'sigmoid':
        return sigmoid_prime
    elif activation_name == 'softmax':
        return None
    else:
        raise ValueError("Activation must be either \"Relu\", \"Sigmoid\" or \"Softmax\"")
