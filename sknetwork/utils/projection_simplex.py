#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 4 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import numpy as np


def projection_simplex(array: np.ndarray, scale: float = 1) -> np.ndarray:
    """
    Project each line of the input onto the Euclidean simplex i.e. solve

    :math:`\\underset{w}{min} ||w - x_i||_2^2` s.t. :math:`\\sum w_j = z, w_j \\ge 0`.

    Parameters
    ----------
    array: np.ndarray
        Data to project. Either one or two dimensional.
    scale: float
        Scale of the simplex i.e. sums of the projected coefficients.

    Returns
    -------
    projection: np.ndarray
        Array with the same shape as the input.

    Example
    -------
    >>> test_array = np.array([[2, 2], [-0.75, 0.25]])
    >>> projection_simplex(test_array)
    array([[0.5, 0.5],
           [0. , 1. ]])

    References
    ----------
    Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T. (2008, July).
    Efficient projections onto the l 1-ball for learning in high dimensions.
    In Proceedings of the 25th international conference on Machine learning (pp. 272-279). ACM.
    http://machinelearning.org/archive/icml2008/papers/361.pdf

    """
    if len(array.shape) == 1:
        array = array.reshape(1, array.shape[0])
    n_samples, n_features = array.shape

    sorted_array = -np.sort(-array)
    cumsum_array = np.cumsum(sorted_array, axis=1) - scale
    denom = 1 + np.arange(n_features)
    condition = sorted_array - cumsum_array / denom > 0
    max_index = np.argmax(condition / denom[::-1], axis=1)
    threshold = (cumsum_array / denom)[np.arange(n_samples), max_index]

    return np.maximum(array - threshold[:, np.newaxis], 0)
