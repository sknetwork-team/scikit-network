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
    matrix : numpy array, sparse CSR matrix or linear operator, shape (n_rows, n_cols)
        Input matrix.
    p :
        Order of the norm.
    Returns
    -------
    norms : np.array, shape (n_rows,)
        Vector norms
    """
    if p == 1:
        norms = matrix.dot(np.ones(matrix.shape[1]))
    elif p == 2:
        if isinstance(matrix, np.ndarray):
            norms = np.linalg.norm(matrix, axis=1)
        elif isinstance(matrix, sparse.csr_matrix):
            data = matrix.data.copy()
            matrix.data = data ** 2
            norms = np.sqrt(matrix.dot(np.ones(matrix.shape[1])))
            matrix.data = data
        else:
            raise NotImplementedError('Norm 2 is not available for a LinearOperator.')
    else:
        raise NotImplementedError('Only norms 1 and 2 are available at the moment.')
    return norms


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
        Normalized matrix.
    """
    norms = get_norms(matrix, p)
    diag = diagonal_pseudo_inverse(norms)
    if hasattr(matrix, 'left_sparse_dot') and callable(matrix.left_sparse_dot):
        return matrix.left_sparse_dot(diag)
    return diag.dot(matrix)
