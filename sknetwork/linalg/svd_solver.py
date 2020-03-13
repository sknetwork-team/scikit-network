#!/usr/bin/env python3
# coding: utf-8
"""
Created on July 10 2019

Authors:
Nathan De Lara <nathan.delara@telecom-paris.fr>
"""
from typing import Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from sknetwork.linalg.randomized_methods import randomized_svd
from sknetwork.linalg.sparse_lowrank import SparseLR
from sknetwork.utils.base import Algorithm


class SVDSolver(Algorithm):
    """
    A generic class for SVD-solvers.

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

    def fit(self, matrix: Union[sparse.csr_matrix, sparse.linalg.LinearOperator, SparseLR], n_components: int):
        """Perform singular value decomposition on input matrix.

        Parameters
        ----------
        matrix:
            Matrix to decompose.
        n_components
            Number of singular values to compute

        Returns
        -------
        self: :class:`SVDSolver`
        """
        return self


class LanczosSVD(SVDSolver):
    """
    An SVD solver using Lanczos method on :math:`AA^T` or :math:`A^TA`.

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

    def __init__(self):
        SVDSolver.__init__(self)

    def fit(self, matrix: Union[sparse.csr_matrix, sparse.linalg.LinearOperator], n_components: int):
        """Perform singular value decomposition on input matrix.

        Parameters
        ----------
        matrix:
            Matrix to decompose.
        n_components
            Number of singular values to compute

        Returns
        -------
        self: :class:`SVDSolver`
        """
        u, s, vt = svds(matrix.astype(np.float), n_components)
        # order the singular values by decreasing order
        index = np.argsort(s)[::-1]
        self.singular_vectors_left_ = u[:, index]
        self.singular_vectors_right_ = vt.T[:, index]
        self.singular_values_ = s[index]

        return self


class HalkoSVD(SVDSolver):
    """
    An SVD solver using Halko's randomized method.

    Parameters
    ----------
    n_oversamples : int (default=10)
        Additional number of random vectors to sample the range of M so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of M is n_components + n_oversamples. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of singular vectors and singular values.
    n_iter : int or 'auto' (default is 'auto')
        See :meth:`randomized_range_finder`
    power_iteration_normalizer : ``'auto'`` (default), ``'QR'``, ``'LU'``, ``None``
        See :meth:`randomized_range_finder`
    transpose : True, False or 'auto' (default)
        Whether the algorithm should be applied to ``matrix.T`` instead of ``matrix``. The
        result should approximately be the same. The 'auto' mode will
        trigger the transposition if ``matrix.shape[1] > matrix.shape[0]`` since this
        implementation of randomized SVD tends to be a little faster in that case.
    flip_sign : boolean, (default=True)
        The output of a singular value decomposition is only unique up to a
        permutation of the signs of the singular vectors. If `flip_sign` is
        set to `True`, the sign ambiguity is resolved by making the largest
        loadings for each component in the left singular vectors positive.
    random_state : int, RandomState instance or None, optional (default=None)
        See :meth:`randomized_range_finder`

    Attributes
    ----------
    singular_vectors_left_: np.ndarray
        Two-dimensional array, each column is a left singular vector of the input.
    singular_vectors_right_: np.ndarray
        Two-dimensional array, each column is a right singular vector of the input.
    singular_values_: np.ndarray
        Singular values.
    """

    def __init__(self, n_oversamples: int = 10, n_iter='auto', transpose='auto',
                 power_iteration_normalizer: Union[str, None] = 'auto', flip_sign: bool = True, random_state=None):
        SVDSolver.__init__(self)
        self.n_oversamples = n_oversamples
        self.n_iter = n_iter
        self.transpose = transpose
        self.power_iteration_normalizer = power_iteration_normalizer
        self.flip_sign = flip_sign
        self.random_state = random_state

    def fit(self, matrix: Union[sparse.csr_matrix, sparse.linalg.LinearOperator, SparseLR], n_components: int):
        """Perform singular value decomposition on input matrix.

        Parameters
        ----------
        matrix:
            Matrix to decompose.
        n_components
            Number of singular values to compute

        Returns
        -------
        self: :class:`SVDSolver`

        """
        u, s, vt = randomized_svd(matrix, n_components, self.n_oversamples, self.n_iter, self.transpose,
                                  self.power_iteration_normalizer, self.flip_sign, self.random_state)
        self.singular_vectors_left_ = u
        self.singular_vectors_right_ = vt.T
        self.singular_values_ = s

        return self
