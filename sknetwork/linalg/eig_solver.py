#!/usr/bin/env python3
# coding: utf-8
"""
Created on July 9 2019
@author: Nathan De Lara <nathan.delara@polytechnique.org>
"""
from abc import ABC
from typing import Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sknetwork.base import Algorithm


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

        * ``'LM'`` : Largest (in modulus) eigenvalues.
        * ``'SM'`` : Smallest (in modulus) eigenvalues.
        * ``'LA'`` : Largest (algebraic) eigenvalues.
        * ``'SA'`` : Smallest (algebraic) eigenvalues.

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

    def fit(self, matrix: Union[sparse.csr_matrix, sparse.linalg.LinearOperator], n_components: int = 2):
        """Perform spectral decomposition on symmetric input matrix.

        Parameters
        ----------
        matrix : sparse.csr_matrix or linear operator
            Matrix to decompose.
        n_components : int
            Number of eigenvectors to compute

        Returns
        -------
        self: :class:`EigSolver`
        """
        self.eigenvalues_, self.eigenvectors_ = eigsh(matrix.astype(float), n_components, which=self.which,
                                                      maxiter=self.n_iter, tol=self.tol)

        return self
