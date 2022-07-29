#!/usr/bin/env python3
# coding: utf-8
"""
Created in April 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""

from typing import Callable

import numpy as np
from scipy import special


def identity(signal: np.ndarray) -> np.ndarray:
    """Apply the identity activation function.
    """

    return signal


def relu(signal: np.ndarray) -> np.ndarray:
    """Apply the Rectified Linear Unit function point-wise:

    :math:`\text{ReLU}(x) = (x)^+ = max(0, x)`
    """

    return np.maximum(signal, 0)


def sigmoid(signal: np.ndarray) -> np.ndarray:
    """Apply the logistic function point-wise:

    :math:`\text{sigmoid}(x) = \frac{1}{(1+e^{-x})}`

    Note: We use the `expit` function from `scipy.special`.
    """

    return special.expit(signal)


def softmax(signal: np.ndarray) -> np.ndarray:
    """Apply the softmax function on each row.

    Note: We use `softmax` function from `scipy.special`.

    Returns
    -------
    np.ndarray
        Output array with same shape as `signal`. Rows will sum to 1.
    """

    return special.softmax(signal, axis=1)


def get_activation_function(activation_name: str) -> Callable[..., np.ndarray]:
    """Returns activation function according to `activation_name`.

    Parameters
    ----------
    activation_name : str
        Which activation function to use. Can be either ``'Identity'``, ``'Relu'``, ``'Sigmoid'`` or ``'Softmax'``.

    Returns
    -------
    Callable[..., np.ndarray]
        Activation function

    Raises
    ------
    ValueError
        Error raised if activation function does not exist.
    """
    activation_name = activation_name.lower()
    if activation_name == 'identity':
        return identity
    elif activation_name == 'relu':
        return relu
    elif activation_name == 'sigmoid':
        return sigmoid
    elif activation_name == 'softmax':
        return softmax
    else:
        raise ValueError("Activation must be either \"Identity\", \"Relu\", \"Sigmoid\" or \"Softmax\"")


def identity_prime(signal: np.ndarray) -> np.ndarray:
    """Derivative of the identity function."""
    return np.ones_like(signal)


def relu_prime(signal: np.ndarray) -> np.ndarray:
    """Derivative of the Rectified Linear Unit function."""
    return 1 * (signal > 0)


def sigmoid_prime(signal: np.ndarray) -> np.ndarray:
    """Derivative of the sigmoid (a.k.a. logistic) function."""
    return sigmoid(signal) * (1 - sigmoid(signal))


def softmax_prime(signal: np.ndarray) -> np.ndarray:
    """Derivative of the softmax function."""
    return np.einsum('ij,jk->ijk', signal, np.eye(signal.shape[-1])) - np.einsum('ij,ik->ijk', signal, signal)


def get_prime_activation_function(activation_name: str) -> Callable[..., np.ndarray]:
    """Returns activation function derivative according to `activation_name`.

    Parameters
    ----------
    activation_name : str
        Which activation function derivative to use. Can be ``'Relu'``, ``'Sigmoid'`` or ``'Softmax'``.

    Returns
    -------
    Callable[..., np.ndarray]
        Activation function

    Raises
    ------
    ValueError
        Error raised if activation function does not exist.
    """
    activation_name = activation_name.lower()
    if activation_name == 'identity':
        return identity_prime
    elif activation_name == 'relu':
        return relu_prime
    elif activation_name == 'sigmoid':
        return sigmoid_prime
    elif activation_name == 'softmax':
        return softmax_prime
    else:
        raise ValueError("Activation must be either \"Identity\", \"Relu\", \"Sigmoid\" or \"Softmax\"")
