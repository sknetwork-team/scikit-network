#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 4 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.linalg.normalization import normalize


def projection_simplex_array(array: np.ndarray, scale: float = 1) -> np.ndarray:
    """Project each line of the input onto the Euclidean simplex i.e. solve

    :math:`\\underset{w}{min} ||w - x_i||_2^2` s.t. :math:`\\sum w_j = z, w_j \\ge 0`.

    Parameters
    ----------
    array: np.ndarray
        Data to project. Either one or two dimensional.
    scale: float
        Scale of the simplex i.e. sums of the projected coefficients.

    Returns
    -------
    projection : np.ndarray
        Array with the same shape as the input.

    Example
    -------
    >>> X = np.array([[2, 2], [-0.75, 0.25]])
    >>> projection_simplex_array(X)
    array([[0.5, 0.5],
           [0. , 1. ]])
    """
    if len(array.shape) == 1:
        array = array.reshape(1, array.shape[0])
    n_row, n_col = array.shape

    sorted_array = -np.sort(-array)
    cumsum_array = np.cumsum(sorted_array, axis=1) - scale
    denom = 1 + np.arange(n_col)
    condition = sorted_array - cumsum_array / denom > 0
    max_index = np.argmax(condition / denom[::-1], axis=1)
    threshold = (cumsum_array / denom)[np.arange(n_row), max_index]

    return np.maximum(array - threshold[:, np.newaxis], 0)


def projection_simplex_csr(matrix: sparse.csr_matrix, scale: float = 1):
    """Project each line of the input onto the Euclidean simplex i.e. solve

    :math:`\\underset{w}{min} ||w - x_i||_2^2` s.t. :math:`\\sum w_j = z, w_j \\ge 0`.

    Parameters
    ----------
    matrix : sparse.csr_matrix
        Matrix whose rows must be projected.
    scale: float
        Scale of the simplex i.e. sums of the projected coefficients.

    Returns
    -------
    projection : sparse.csr_matrix
        Matrix with the same shape as the input.

    Examples
    --------
    >>> X = sparse.csr_matrix(np.array([[2, 2], [-0.75, 0.25]]))
    >>> X_proj = projection_simplex_csr(X)
    >>> X_proj.nnz
    3
    >>> X_proj.toarray()
    array([[0.5, 0.5],
           [0. , 1. ]])
    """
    data = matrix.data
    if data.dtype == bool or (data.min() == data.max()):
        return normalize(matrix, p=1)

    indptr = matrix.indptr
    new_data = np.zeros_like(data)

    for i in range(indptr.size-1):
        j1 = indptr[i]
        j2 = indptr[i+1]
        new_data[j1:j2] = projection_simplex_array(data[j1:j2], scale=scale)

    new_matrix = sparse.csr_matrix((new_data, matrix.indices, indptr), shape=matrix.shape)
    new_matrix.eliminate_zeros()
    return new_matrix


def projection_simplex(x: Union[np.ndarray, sparse.csr_matrix], scale: float = 1.):
    """Project each line of the input onto the Euclidean simplex i.e. solve

    :math:`\\underset{w}{min} ||w - x_i||_2^2` s.t. :math:`\\sum w_j = z, w_j \\ge 0`.

    Parameters
    ----------
    x :
        Data to project. Either one or two dimensional. Can be sparse or dense.
    scale : float
        Scale of the simplex i.e. sums of the projected coefficients.

    Returns
    -------
    projection : np.ndarray or sparse.csr_matrix
        Array with the same type and shape as the input.

    Example
    -------
    >>> X = np.array([[2, 2], [-0.75, 0.25]])
    >>> projection_simplex(X)
    array([[0.5, 0.5],
           [0. , 1. ]])
    >>> X_csr = sparse.csr_matrix(X)
    >>> X_proj = projection_simplex(X_csr)
    >>> X_proj.nnz
    3
    >>> X_proj.toarray()
    array([[0.5, 0.5],
           [0. , 1. ]])

    References
    ----------
    Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T. (2008, July).
    `Efficient projections onto the l 1-ball for learning in high dimensions.
    <http://machinelearning.org/archive/icml2008/papers/361.pdf>`_
    In Proceedings of the 25th international conference on Machine learning (pp. 272-279). ACM.
    """
    if isinstance(x, np.ndarray):
        return projection_simplex_array(x, scale)
    elif isinstance(x, sparse.csr_matrix):
        return projection_simplex_csr(x, scale)
    else:
        raise TypeError('Input must be a numpy array or a CSR matrix.')
