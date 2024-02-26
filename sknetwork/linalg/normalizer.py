#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in November 2019
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""
from typing import Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator


def diagonal_pseudo_inverse(weights: np.ndarray) -> sparse.csr_matrix:
    """Compute :math:`\\text{diag}(w)^+`, the pseudo-inverse of the diagonal matrix
    with diagonal elements given by the weights :math:`w`.

    Parameters
    ----------
    weights:
        The weights to invert.

    Returns
    -------
    sparse.csr_matrix

    """
    diag: sparse.csr_matrix = sparse.diags(weights, format='csr')
    diag.data = 1 / diag.data
    return diag


def get_norms(matrix: Union[sparse.csr_matrix, np.ndarray, LinearOperator], p=1):
    """Get the norms of rows of a matrix.

    Parameters
    ----------
    matrix : numpy array or sparse CSR matrix or LinearOperator, shape (n_rows, n_cols)
        Input matrix.
    p :
        Order of the norm (1 or 2).
    Returns
    -------
    norms : np.array, shape (n_rows,)
        Vector norms
    """
    n_row, n_col = matrix.shape
    if isinstance(matrix, np.ndarray):
        input_matrix = sparse.csr_matrix(matrix)
    elif isinstance(matrix, sparse.csr_matrix):
        input_matrix = matrix.copy()
    else:
        input_matrix = matrix
    if p == 1:
        if not isinstance(matrix, LinearOperator):
            input_matrix.data = np.abs(input_matrix.data)
        return input_matrix.dot(np.ones(n_col))
    elif p == 2:
        if isinstance(matrix, LinearOperator):
            raise ValueError('Only norm 1 is available for linear operators.')
        input_matrix.data = input_matrix.data**2
        return np.sqrt(input_matrix.dot(np.ones(n_col)))
    else:
        raise ValueError('Only norms 1 and 2 are available.')


def normalize(matrix: Union[sparse.csr_matrix, np.ndarray, LinearOperator], p=1):
    """Normalize the rows of a matrix so that all have norm 1 (or 0; null rows remain null).

    Parameters
    ----------
    matrix :
        Input matrix.
    p :
        Order of the norm.

    Returns
    -------
    normalized matrix :
        Normalized matrix (same format as input matrix).
    """
    norms = get_norms(matrix, p)
    diag = diagonal_pseudo_inverse(norms)
    if hasattr(matrix, 'left_sparse_dot') and callable(matrix.left_sparse_dot):
        return matrix.left_sparse_dot(diag)
    return diag.dot(matrix)
