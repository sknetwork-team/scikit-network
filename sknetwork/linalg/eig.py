#!/usr/bin/env python3
# coding: utf-8
"""
Created on July 9 2019

Authors:
Nathan De Lara <nathan.delara@telecom-paris.fr>
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sknetwork.utils.algorithm_base_class import Algorithm
from sknetwork.linalg import SparseLR
from sknetwork.linalg.randomized_matrix_factorization import randomized_eig
from typing import Union


class EigSolver(Algorithm):
    """
    A generic class for eigensolvers.

    Parameters
    ----------
    which: str
        Which eigenvectors and eigenvalues to find:

        * ``'LM'`` : Largest (in magnitude) eigenvalues.
        * ``'SM'` : Smallest (in magnitude) eigenvalues.

    Attributes
    ----------
    eigenvectors_: np.ndarray
        Two-dimensional array, each column is an eigenvector of the input.
    eigenvalues_: np.ndarray
        Eigenvalues associated to each eigenvector.
    """

    def __init__(self, which='LM'):
        self.which = which

        self.eigenvectors_ = None
        self.eigenvalues_ = None

    def fit(self, matrix: Union[sparse.csr_matrix, sparse.linalg.LinearOperator, SparseLR], n_components: int):
        """Perform eigenvalue decomposition on input matrix.

        Parameters
        ----------
        matrix:
            Matrix to decompose.
        n_components
            Number of eigenvectors to compute

        Returns
        -------
        self: :class:`EigSolver`

        """
        return self


class LanczosEig(EigSolver):
    """
    An eigenvalue solver using Lanczos method.

    Parameters
    ----------
    which: str
        Which eigenvectors and eigenvalues to find:

        * ``'LM'`` : Largest (in magnitude) eigenvalues.
        * ``'SM'`` : Smallest (in magnitude) eigenvalues.

    Attributes
    ----------
    eigenvectors_: np.ndarray
        Two-dimensional array, each column is an eigenvector of the input.
    eigenvalues_: np.ndarray
        Eigenvalues associated to each eigenvector.

    See Also
    --------
    scipy.sparse.linalg.eigsh

    """

    def __init__(self, which):
        EigSolver.__init__(self, which=which)

    def fit(self, matrix: Union[sparse.csr_matrix, sparse.linalg.LinearOperator], n_components: int):
        """Perform eigenvalue decomposition on input matrix.

        Parameters
        ----------
        matrix:
            Matrix to decompose.
        n_components
            Number of eigenvectors to compute

        Returns
        -------
        self: :class:`EigSolver`

        """
        eigenvalues, eigenvectors = eigsh(matrix.astype(np.float), n_components, which=self.which)
        self.eigenvectors_ = eigenvectors
        self.eigenvalues_ = eigenvalues

        return self


class HalkoEig(EigSolver):
    """
    An eigenvalue solver using Halko's randomized method.

    Parameters
    ----------
    which: str
        Which eigenvectors and eigenvalues to find:

        * ``'LM'`` : Largest (in magnitude) eigenvalues.
        * ``'SM'`` : Smallest (in magnitude) eigenvalues.

    n_oversamples : int (default=10)
        Additional number of random vectors to sample the range of ``matrix`` so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of ``matrix`` is ``n_components + n_oversamples``. Smaller number can improve speed
        but can negatively impact the quality of approximation of singular vectors and singular values.
    n_iter: int or 'auto' (default is 'auto')
        See :meth:`randomized_range_finder`
    power_iteration_normalizer: ``'auto'`` (default), ``'QR'``, ``'LU'``, ``None``
        See :meth:`randomized_range_finder`
    random_state: int, RandomState instance or None, optional (default=None)
        See :meth:`randomized_range_finder`
    one_pass: bool (default=False)
        whether to use algorithm 5.6 instead of 5.3. 5.6 requires less access to the original matrix,
        while 5.3 is more accurate.

    """

    def __init__(self, which='LM', n_oversamples: int = 10, n_iter='auto',
                 power_iteration_normalizer: Union[str, None] = 'auto', random_state=None, one_pass: bool = False):
        EigSolver.__init__(self, which=which)
        self.n_oversamples = n_oversamples
        self.n_iter = n_iter
        self.power_iteration_normalizer = power_iteration_normalizer
        self.random_state = random_state
        self.one_pass = one_pass

    def fit(self, matrix: Union[sparse.csr_matrix, sparse.linalg.LinearOperator, SparseLR], n_components: int):
        """Perform eigenvalue decomposition on input matrix.

        Parameters
        ----------
        matrix :
            Matrix to decompose.
        n_components :
            Number of eigenvectors to compute

        Returns
        -------
        self: :class:`EigSolver`

        """
        eigenvalues, eigenvectors = randomized_eig(matrix, n_components, self.which, self.n_oversamples, self.n_iter,
                                                   self.power_iteration_normalizer, self.random_state, self.one_pass)
        self.eigenvectors_ = eigenvectors
        self.eigenvalues_ = eigenvalues

        return self
