#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
from typing import Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator


def diag_pinv(weights: np.ndarray) -> sparse.csr_matrix:
    """Compute :math:`W^+ = \\text{diag}(w)^+`, the pseudo inverse of the diagonal matrix
    with diagonal the weights :math:`w`.

    Parameters
    ----------
    weights:
        The weights to invert.

    Returns
    -------
    sparse.csr_matrix
        :math:`W^+`

    """
    diag: sparse.csr_matrix = sparse.diags(weights, format='csr')
    diag.data = 1 / diag.data
    return diag


def normalize(matrix: Union[sparse.csr_matrix, np.ndarray, LinearOperator], p=1):
    """Normalize rows of a matrix. Null rows remain null.

    Parameters
    ----------
    matrix :
        Input matrix.
    p :
        Order of the norm

    Returns
    -------
    normalized matrix : same as input
    """
    if p == 1:
        norm = matrix.dot(np.ones(matrix.shape[1]))
    elif p == 2:
        if isinstance(matrix, np.ndarray):
            norm = np.linalg.norm(matrix, axis=1)
        elif isinstance(matrix, sparse.csr_matrix):
            data = matrix.data.copy()
            matrix.data = data ** 2
            norm = np.sqrt(matrix.dot(np.ones(matrix.shape[1])))
            matrix.data = data
        else:
            raise NotImplementedError('Norm 2 is not available for LinearOperator.')
    else:
        raise NotImplementedError('Only norms 1 and 2 are available at the moment.')

    diag = diag_pinv(norm)
    if hasattr(matrix, 'left_sparse_dot') and callable(matrix.left_sparse_dot):
        return matrix.left_sparse_dot(diag)
    return diag.dot(matrix)
