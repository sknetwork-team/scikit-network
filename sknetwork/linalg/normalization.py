#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
from typing import Union

import numpy as np
from scipy import sparse


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


def normalize(matrix: Union[sparse.csr_matrix, np.ndarray]) -> sparse.csr_matrix:
    """Normalize a matrix so that rows sum to 1 (or 0).

    Parameters
    ----------
    matrix :
        Input matrix.

    Returns
    -------
    New matrix.

    """
    diag = diag_pinv(matrix.dot(np.ones(matrix.shape[1])))
    return diag.dot(matrix)
