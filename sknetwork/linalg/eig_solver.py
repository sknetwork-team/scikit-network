#!/usr/bin/env python3
# coding: utf-8
"""
Created on July 9 2019
@author: Nathan De Lara <ndelara@enst.fr>
"""
from abc import ABC
from typing import Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sknetwork.utils.base import Algorithm


class EigSolver(Algorithm, ABC):
    """Generic class for eigensolvers.

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


class LanczosEig(EigSolver):
    """Eigenvalue solver using Lanczos method.

    Parameters
    ----------
    which : str
        Which eigenvectors and eigenvalues to find:

        * ``'LM'`` : Largest (in magnitude) eigenvalues.
        * ``'SM'`` : Smallest (in magnitude) eigenvalues.
    n_iter : int
        Maximum number of Arnoldi update iterations allowed.
        Default = 10 * nb of rows.
    tol : float
        Relative accuracy for eigenvalues (stopping criterion).
        Default = 0 (machine precision).
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
    def __init__(self, which='LM', n_iter: int = None, tol: float = 0.):
        super(LanczosEig, self).__init__(which=which)
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, matrix: Union[sparse.csr_matrix, sparse.linalg.LinearOperator], n_components: int,
            init_vector: np.ndarray = None):
        """Perform eigenvalue decomposition on symmetric input matrix.

        Parameters
        ----------
        matrix :
            Matrix to decompose.
        n_components : int
            Number of eigenvectors to compute
        init_vector : np.ndarray
            Starting vector for iteration. Default: random.
        Returns
        -------
        self: :class:`EigSolver`
        """
        eigenvalues, eigenvectors = eigsh(matrix.astype(float), n_components, which=self.which, maxiter=self.n_iter,
                                          tol=self.tol, v0=init_vector)
        self.eigenvectors_ = eigenvectors
        self.eigenvalues_ = eigenvalues

        if self.which in ['LM', 'LA']:
            index = np.argsort(-self.eigenvalues_)
            self.eigenvalues_ = self.eigenvalues_[index]
            self.eigenvectors_ = self.eigenvectors_[:, index]

        return self
