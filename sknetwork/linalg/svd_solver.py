#!/usr/bin/env python3
# coding: utf-8
"""
Created on July 10 2019

Authors:
Nathan De Lara <nathan.delara@telecom-paris.fr>
"""
from abc import ABC
from typing import Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from sknetwork.base import Algorithm


class SVDSolver(Algorithm, ABC):
    """Generic class for SVD-solvers.

    Attributes
    ----------
    singular_vectors_left_: np.ndarray
        Two-dimensional array, each column is a left singular vector of the input.
    singular_vectors_right_: np.ndarray
        Two-dimensional array, each column is a right singular vector of the input.
    singular_values_: np.ndarray
        Singular values.
    """
    def __init__(self):
        self.singular_vectors_left_ = None
        self.singular_vectors_right_ = None
        self.singular_values_ = None


class LanczosSVD(SVDSolver):
    """SVD solver using Lanczos method on :math:`AA^T` or :math:`A^TA`.

    Parameters
    ----------
    n_iter : int
        Maximum number of Arnoldi update iterations allowed.
        Default = 10 * nb or rows or columns.
    tol : float
        Relative accuracy for eigenvalues (stopping criterion).
        Default = 0 (machine precision).

    Attributes
    ----------
    singular_vectors_left_: np.ndarray
        Two-dimensional array, each column is a left singular vector of the input.
    singular_vectors_right_: np.ndarray
        Two-dimensional array, each column is a right singular vector of the input.
    singular_values_: np.ndarray
        Singular values.

    See Also
    --------
    scipy.sparse.linalg.svds
    """
    def __init__(self, n_iter: int = None, tol: float = 0.):
        super(LanczosSVD, self).__init__()
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, matrix: Union[sparse.csr_matrix, sparse.linalg.LinearOperator], n_components: int,
            init_vector: np.ndarray = None):
        """Perform singular value decomposition on input matrix.

        Parameters
        ----------
        matrix :
            Matrix to decompose.
        n_components : int
            Number of singular values to compute
        init_vector : np.ndarray
            Starting vector for iteration.
            Default = random.
        Returns
        -------
        self: :class:`SVDSolver`
        """
        u, s, vt = svds(matrix.astype(float), n_components, v0=init_vector)
        # order the singular values by decreasing order
        index = np.argsort(-s)
        self.singular_vectors_left_ = u[:, index]
        self.singular_vectors_right_ = vt.T[:, index]
        self.singular_values_ = s[index]

        return self
