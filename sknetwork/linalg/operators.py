#!/usr/bin/env python3
# coding: utf-8
"""
Created in April 2020
@author: Thomas Bonald <bonald@enst.fr>
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""
from typing import Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from sknetwork.linalg import diagonal_pseudo_inverse
from sknetwork.linalg.normalizer import normalize
from sknetwork.linalg.sparse_lowrank import SparseLR
from sknetwork.utils.check import check_format


class Regularizer(SparseLR):
    """Regularized matrix as a Scipy LinearOperator.

    Defined by :math:`A + \\alpha \\frac{11^T}n` where :math:`A` is the input matrix
    and :math:`\\alpha` the regularization factor.

    Parameters
    ----------
    input_matrix :
        Input matrix.
    regularization : float
        Regularization factor.
        Default value = 1.

    Examples
    --------
    >>> from sknetwork.data import house
    >>> adjacency = house()
    >>> regularizer = Regularizer(adjacency)
    >>> regularizer.dot(np.ones(5))
    array([3., 4., 3., 3., 4.])
    """
    def __init__(self, input_matrix: Union[sparse.csr_matrix, np.ndarray], regularization: float = 1):
        n_row, n_col = input_matrix.shape
        u = regularization * np.ones(n_row)
        v = np.ones(n_col) / n_col
        super(Regularizer, self).__init__(input_matrix, (u, v))


class Normalizer(LinearOperator):
    """Normalized matrix as a Scipy LinearOperator.

    Defined by :math:`D^{-1}A` where :math:`A` is the regularized adjacency matrix and :math:`D` the corresponding
    diagonal matrix of degrees (sums over rows).

    Parameters
    ----------
    adjacency :
        :term:`Adjacency <adjacency>` matrix of the graph.
    regularization : float
        Regularization factor.
        Default value = 0.

    Examples
    --------
    >>> from sknetwork.data import house
    >>> adjacency = house()
    >>> normalizer = Normalizer(adjacency)
    >>> normalizer.dot(np.ones(5))
    array([1., 1., 1., 1., 1.])
    """
    def __init__(self, adjacency: Union[sparse.csr_matrix, np.ndarray], regularization: float = 0):
        if adjacency.ndim == 1:
            adjacency = adjacency.reshape(1, -1)
        super(Normalizer, self).__init__(dtype=float, shape=adjacency.shape)
        n_col = adjacency.shape[1]
        self.regularization = regularization
        self.adjacency = adjacency
        self.norm_diag = diagonal_pseudo_inverse(adjacency.dot(np.ones(n_col)) + regularization)

    def _matvec(self, matrix: np.ndarray):
        prod = self.adjacency.dot(matrix)
        if self.regularization > 0:
            n_row = self.shape[0]
            if matrix.ndim == 2:
                prod += self.regularization * np.outer(np.ones(n_row), matrix.mean(axis=0))
            else:
                prod += self.regularization * matrix.mean() * np.ones(n_row)
        return self.norm_diag.dot(prod)

    def _transpose(self):
        return self


class Laplacian(LinearOperator):
    """Laplacian matrix as a Scipy LinearOperator.

    Defined by :math:`L = D - A` where :math:`A` is the regularized adjacency matrix and :math:`D` the corresponding
    diagonal matrix of degrees.

    If normalized, defined by :math:`L = I - D^{-1/2}AD^{-1/2}`.

    Parameters
    ----------
    adjacency :
        :term:`Adjacency <adjacency>` matrix of the graph.
    regularization : float
        Regularization factor.
        Default value = 0.
    normalized_laplacian : bool
        If ``True``, use normalized Laplacian.
        Default value = ``False``.

    Examples
    --------
    >>> from sknetwork.data import house
    >>> adjacency = house()
    >>> laplacian = Laplacian(adjacency)
    >>> laplacian.dot(np.ones(5))
    array([0., 0., 0., 0., 0.])
    """
    def __init__(self, adjacency: Union[sparse.csr_matrix, np.ndarray], regularization: float = 0,
                 normalized_laplacian: bool = False):
        super(Laplacian, self).__init__(dtype=float, shape=adjacency.shape)
        n = adjacency.shape[0]
        self.regularization = regularization
        self.normalized_laplacian = normalized_laplacian
        self.weights = adjacency.dot(np.ones(n))
        self.laplacian = sparse.diags(self.weights, format='csr') - adjacency
        if self.normalized_laplacian:
            self.norm_diag = diagonal_pseudo_inverse(np.sqrt(self.weights + regularization))

    def _matvec(self, matrix: np.ndarray):
        if self.normalized_laplacian:
            matrix = self.norm_diag.dot(matrix)
        prod = self.laplacian.dot(matrix)
        if self.regularization > 0:
            n = self.shape[0]
            if matrix.ndim == 2:
                prod += self.regularization * (matrix - np.outer(np.ones(n), matrix.mean(axis=0)))
            else:
                prod += self.regularization * (matrix - matrix.mean())
        if self.normalized_laplacian:
            prod = self.norm_diag.dot(prod)
        return prod

    def _transpose(self):
        return self

    def astype(self, dtype: Union[str, np.dtype]):
        """Change dtype of the object."""
        self.dtype = np.dtype(dtype)
        self.laplacian = self.laplacian.astype(self.dtype)
        return self


class CoNeighbor(LinearOperator):
    """Co-neighborhood adjacency as a LinearOperator.

    :math:`\\tilde{A} = AF^{-1}A^T`, or :math:`\\tilde{B} = BF^{-1}B^T`.

    where F is a weight matrix.

    Parameters
    ----------
    adjacency:
        Adjacency or biadjacency of the input graph.
    normalized:
        If ``True``, F is the diagonal in-degree matrix :math:`F = \\text{diag}(A^T1)`.
        Otherwise, F is the identity matrix.

    Examples
    --------
    >>> from sknetwork.data import star_wars
    >>> biadjacency = star_wars(metadata=False)
    >>> d_out = biadjacency.dot(np.ones(3))
    >>> coneighbor = CoNeighbor(biadjacency)
    >>> np.allclose(d_out, coneighbor.dot(np.ones(4)))
    True
    """
    def __init__(self, adjacency: Union[sparse.csr_matrix, np.ndarray], normalized: bool = True):
        adjacency = check_format(adjacency).astype(float)
        n = adjacency.shape[0]
        super(CoNeighbor, self).__init__(dtype=float, shape=(n, n))

        if normalized:
            self.forward = normalize(adjacency.T).tocsr()
        else:
            self.forward = adjacency.T

        self.backward = adjacency

    def __neg__(self):
        self.backward *= -1
        return self

    def __mul__(self, other):
        self.backward *= other
        return self

    def _matvec(self, matrix: np.ndarray):
        return self.backward.dot(self.forward.dot(matrix))

    def _transpose(self):
        """Transposed operator"""
        operator = CoNeighbor(self.backward)
        operator.backward = self.forward.T.tocsr()
        operator.forward = self.backward.T.tocsr()
        return operator

    def left_sparse_dot(self, matrix: sparse.csr_matrix):
        """Left dot product with a sparse matrix"""
        self.backward = matrix.dot(self.backward)
        return self

    def right_sparse_dot(self, matrix: sparse.csr_matrix):
        """Right dot product with a sparse matrix"""
        self.forward = self.forward.dot(matrix)
        return self

    def astype(self, dtype: Union[str, np.dtype]):
        """Change dtype of the object."""
        self.backward.astype(dtype)
        self.forward.astype(dtype)
        self.dtype = dtype
        return self
