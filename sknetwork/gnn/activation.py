#!/usr/bin/env python3
# coding: utf-8
"""
Created on Thu Apr 21 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""

import numpy as np
import scipy


def relu(input: np.ndarray) -> np.ndarray:
    """Applies Rectified Linear Unit function:

    :math:`\text{ReLU}(x) = (x)^+ = max(0, x)`

    Parameters
    ----------
    input : np.ndarray (shape (n_samples, n_features))
        input array.

    Returns
    -------
    np.ndarray (shape (n_samples, n_features))
        output array.
    """

    return np.maximum(input, 0)


def tanh(input: np.ndarray) -> np.ndarray:
    """Applies hyperbolic tan function.

    Parameters
    ----------
    input : np.ndarray (shape (n_samples, n_features))
        input array.

    Returns
    -------
    np.ndarray (shape (n_samples, n_features))
        output array.
    """

    return np.tanh(input)


def sigmoid(input: np.ndarray) -> np.ndarray:
    """Compute logistic function.

    Note: We use `expit` function from `scipy.special`.

    Parameters
    ----------
    input : np.ndarray (shape (n_samples, n_features))
        input array.

    Returns
    -------
    np.ndarray (shape (n_samples, n_features))
        output array.
    """

    return scipy.special.expit(input)


def softmax(input: np.ndarray) -> np.ndarray:
    """Compute softmax function.

    Note: We use `softmax` function from `scipy.special`.

    Parameters
    ----------
    input : np.ndarray
        input array.

    Returns
    -------
    np.ndarray
        output array with same shape as `input`. Rows will sum to 1.
    """

    return scipy.special.softmax(input, axis=1)


ACTIVATIONS = {
    'tanh': tanh,
    'relu': relu,
    'sigmoid': sigmoid,
    'softmax': softmax
}


def relu_prime(input: np.ndarray) -> np.ndarray:
    """Derivative of Rectified Linear Unit function.

    Parameters
    ----------
    input : np.ndarray
        input array.

    Returns
    -------
    np.ndarray
        output array.
    """

    return 1 * (input > 0)


def sigmoid_prime(input: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid function.

    Parameters
    ----------
    input : np.ndarray
        input array.

    Returns
    -------
    np.ndarray
        output array.
    """

    return sigmoid(input) * (1 - sigmoid(input))


def softmax_prime(input: np.ndarray) -> np.ndarray:
    """Derivative of softmax function.

    Parameters
    ----------
    input : np.ndarray
        input array.

    Returns
    -------
    np.ndarray
        output array.
    """
    pass


DERIVATIVES = {
    'sigmoid': sigmoid_prime,
    'softmax': softmax_prime,
    'relu': relu_prime
}
