#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 2020
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator


def safe_sparse_dot(a, b):
    """Dot product with proper use of the sparse matrix format.
    Use BLAS instead of numpy.dot when possible to avoid unnecessary copies.

    Parameters
    ----------
    a : array, sparse matrix or LinearOperator
    b : array, sparse matrix or LinearOperator
    Returns
    -------
    dot_product : array or sparse matrix
        sparse if ``a`` or ``b`` is sparse.
    """
    if type(a) == np.ndarray:
        return b.T.dot(a.T).T
    if isinstance(a, LinearOperator) and isinstance(b, LinearOperator):
        raise NotImplementedError
    if hasattr(a, 'right_sparse_dot') and type(b) == sparse.csr_matrix:
        if callable(a.right_sparse_dot):
            return a.right_sparse_dot(b)
    if hasattr(b, 'left_sparse_dot') and type(a) == sparse.csr_matrix:
        if callable(b.left_sparse_dot):
            return b.left_sparse_dot(a)
    else:
        return a.dot(b)
