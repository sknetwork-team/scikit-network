#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""

from typing import Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from sknetwork.utils.check import check_format, check_square


class Polynome(LinearOperator):
    """Polynome of an adjacency matrix as a linear operator

    :math:`P(A) = \\alpha_k A^k + ... + \\alpha_1 A + \\alpha_0`.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph
    coeffs : np.ndarray
        Coefficients of the polynome by increasing order of power.

    Examples
    --------
    >>> from scipy import sparse
    >>> from sknetwork.linalg import Polynome
    >>> adjacency = sparse.eye(2, format='csr')
    >>> polynome = Polynome(adjacency, np.arange(3))
    >>> x = np.ones(2)
    >>> polynome.dot(x)
    array([3., 3.])
    >>> polynome.T.dot(x)
    array([3., 3.])

    Notes
    -----
    The polynome is evaluated using the `Ruffini-Horner method
    <https://en.wikipedia.org/wiki/Horner%27s_method>`_.
    """

    def __init__(self, adjacency: Union[sparse.csr_matrix, np.ndarray], coeffs: np.ndarray):
        if coeffs.shape[0] == 0:
            raise ValueError('A polynome requires at least one coefficient.')
        if not isinstance(adjacency, LinearOperator):
            adjacency = check_format(adjacency)
        check_square(adjacency)
        shape = adjacency.shape
        dtype = adjacency.dtype
        super(Polynome, self).__init__(dtype=dtype, shape=shape)

        self.adjacency = adjacency
        self.coeffs = coeffs

    def __neg__(self):
        return Polynome(self.adjacency, -self.coeffs)

    def __mul__(self, other):
        return Polynome(self.adjacency, other * self.coeffs)

    def _matvec(self, matrix: np.ndarray):
        """Right dot product with a dense matrix.
        """
        y = self.coeffs[-1] * matrix
        for a in self.coeffs[::-1][1:]:
            y = self.adjacency.dot(y) + a * matrix
        return y

    def _transpose(self):
        """Transposed operator."""
        return Polynome(self.adjacency.T.tocsr(), self.coeffs)
