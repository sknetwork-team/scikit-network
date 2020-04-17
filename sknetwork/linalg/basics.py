#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""

import numpy as np
from scipy import sparse

from sknetwork.linalg.sparse_lowrank import SparseLR


def safe_sparse_dot(a, b):
    """Dot product that handles the sparse matrix case correctly.
    Uses BLAS GEMM as replacement for numpy.dot where possible to avoid unnecessary copies.

    Parameters
    ----------
    a : array or sparse matrix or SparseLR
    b : array or sparse matrix or SparseLR
    Returns
    -------
    dot_product : array or sparse matrix
        sparse if ``a`` or ``b`` is sparse.
    """
    if type(a) == SparseLR and type(b) == np.ndarray:
        return a.dot(b)
    if type(b) == SparseLR and type(a) == np.ndarray:
        return b.T.dot(a.T).T
    if type(a) == SparseLR and type(b) == SparseLR:
        raise NotImplementedError
    if type(a) == SparseLR and type(b) == sparse.csr_matrix:
        return a.right_sparse_dot(b)
    if type(b) == SparseLR and type(a) == sparse.csr_matrix:
        return b.left_sparse_dot(a)
    if type(a) == np.ndarray:
        return b.T.dot(a.T).T
    return a.dot(b)
