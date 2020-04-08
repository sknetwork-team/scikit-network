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


def normalize(matrix: Union[sparse.csr_matrix, np.ndarray, LinearOperator], p=1) -> sparse.csr_matrix:
    """Normalize a matrix so that rows sum to 1 (or 0).

    Parameters
    ----------
    matrix :
        Input matrix.
    p :
        Order of the norm

    Returns
    -------
    New matrix.

    """
    if p == 1:
        norm = matrix.dot(np.ones(matrix.shape[1]))
    elif p == 2 and isinstance(matrix, np.ndarray):
        norm = np.linalg.norm(matrix, axis=1)
    else:
        raise NotImplementedError('Only norms 1 and 2 are available at the moment.'
                                  'Norm 2 is only available for Numpy arrays.')

    diag = diag_pinv(norm)
    if hasattr(matrix, 'left_sparse_dot'):
        return matrix.left_sparse_dot(diag)
    return diag.dot(matrix)
