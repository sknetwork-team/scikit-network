#!/usr/bin/env python3
# coding: utf-8
"""
Created on Thu Apr 21 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""

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


ACTIVATIONS = {
    'relu': relu,
    'sigmoid': sigmoid,
    'softmax': softmax
}


def relu_prime(input: np.ndarray) -> np.ndarray:
    """Derivative of the Rectified Linear Unit function."""
    return 1 * (input > 0)


def sigmoid_prime(input: np.ndarray) -> np.ndarray:
    """Derivative of the sigmoid (a.k.a. logistic) function."""
    return sigmoid(input) * (1 - sigmoid(input))


DERIVATIVES = {
    'sigmoid': sigmoid_prime,
    'softmax': None,
    'relu': relu_prime
}
