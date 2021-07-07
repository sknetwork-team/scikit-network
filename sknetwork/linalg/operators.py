#!/usr/bin/env python3
# coding: utf-8
"""
Created on Apr 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""
from typing import Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from sknetwork.linalg.normalization import normalize
from sknetwork.linalg.sparse_lowrank import SparseLR
from sknetwork.utils.check import check_format


class RegularizedAdjacency(SparseLR):
    """Regularized adjacency matrix as a Scipy LinearOperator.

    Defined by :math:`A + \\alpha \\frac{11^T}n` where :math:`A` is the adjacency matrix
    and :math:`\\alpha` the regularization factor.

    Parameters
    ----------
    adjacency :
        :term:`Adjacency <adjacency>` matrix of the graph.
    regularization : float
        Regularization factor `\\alpha`.
        Default value = 1.

    Examples
    --------
    >>> from sknetwork.data import house
    >>> adjacency = house()
    >>> adjacency_reg = RegularizedAdjacency(adjacency)
    >>> adjacency_reg.dot(np.ones(5))
    array([3., 4., 3., 3., 4.])
    """
    def __init__(self, adjacency: Union[sparse.csr_matrix, np.ndarray], regularization: float = 1):
        n_row, n_col = adjacency.shape
        x = regularization * np.ones(n_row)
        y = np.ones(n_col) / n_col
        super(RegularizedAdjacency, self).__init__(adjacency, (x, y))


class RegularizedLaplacian(LinearOperator):
    """Regularized Laplacian matrix as a Scipy LinearOperator.

    Defined by :math:`L = D - A` where :math:`A` is the regularized adjacency matrix and :math:`D` the corresponding
    diagonal matrix of degrees.

    Parameters
    ----------
    adjacency :
        :term:`Adjacency <adjacency>` matrix of the graph.
    regularization : float
        Regularization factor.
        Default value = 1.

    Examples
    --------
    >>> from sknetwork.data import house
    >>> adjacency = house()
    >>> laplacian = RegularizedLaplacian(adjacency)
    >>> laplacian.dot(np.ones(adjacency.shape[1]))
    array([0., 0., 0., 0., 0.])
    """
    def __init__(self, adjacency: Union[sparse.csr_matrix, np.ndarray], regularization: float = 1):
        super(RegularizedLaplacian, self).__init__(dtype=float, shape=adjacency.shape)
        n = adjacency.shape[0]
        self.regularization = regularization
        self.weights = adjacency.dot(np.ones(n)) + regularization
        self.laplacian = sparse.diags(self.weights, format='csr') - adjacency

    def _matvec(self, matrix: np.ndarray):
        prod = self.laplacian.dot(matrix)
        n = self.shape[0]
        if matrix.ndim == 2:
            prod -= self.regularization * np.tile(matrix.sum(axis=0), (n, 1)) / n
        else:
            prod -= self.regularization * matrix.sum() / n
        return prod

    def _transpose(self):
        return self

    def astype(self, dtype: Union[str, np.dtype]):
        """Change dtype of the object."""
        self.dtype = np.dtype(dtype)
        self.laplacian = self.laplacian.astype(self.dtype)
        self.weights = self.weights.astype(self.dtype)

        return self


class NormalizedRegularizedLaplacian(LinearOperator):
    """Normalized regularized Laplacian matrix as a Scipy LinearOperator.

    Defined by :math:`L = I - D^{-1/2}AD^{-1/2}` where :math:`A` is the regularized adjacency matrix
    and :math:`D` the corresponding diagonal matrix of degrees.

    Parameters
    ----------
    adjacency :
        :term:`Adjacency <adjacency>` matrix of the graph.
    regularization : float
        Regularization factor.
        Default value = 1.

    Examples
    --------
    >>> from sknetwork.data import house
    >>> adjacency = house()
    >>> laplacian = NormalizedRegularizedLaplacian(adjacency, regularization=0)
    >>> degrees = adjacency.dot(np.ones(5))
    >>> all(np.abs(laplacian.dot(np.sqrt(degrees))) < 1e-10)
    True
    """
    def __init__(self, adjacency: Union[sparse.csr_matrix, np.ndarray], regularization: float = 1):
        super(NormalizedRegularizedLaplacian, self).__init__(dtype=float, shape=adjacency.shape)
        n = adjacency.shape[0]
        self.regularization = regularization
        weights = adjacency.dot(np.ones(n)) + regularization
        weights_inv = np.zeros(n)
        weights_inv[weights > 0] = 1 / weights[weights > 0]
        self.weights_inv_sqrt = np.sqrt(weights_inv)
        self.weights_inv_sqrt_vec = self.weights_inv_sqrt.reshape(-1, 1)
        weights_inv_sqrt_diag = sparse.diags(self.weights_inv_sqrt)
        laplacian = sparse.diags(weights, format='csr') - adjacency
        self.normalized_laplacian = weights_inv_sqrt_diag.dot(laplacian).dot(weights_inv_sqrt_diag)

    def _matvec(self, matrix: np.ndarray):
        prod = self.normalized_laplacian.dot(matrix)
        n = self.shape[0]
        if matrix.ndim == 2:
            prod -= self.regularization * self.weights_inv_sqrt_vec.dot(self.weights_inv_sqrt_vec.T.dot(matrix)) / n
        else:
            prod -= self.regularization * self.weights_inv_sqrt * np.inner(self.weights_inv_sqrt, matrix) / n
        return prod

    def _transpose(self):
        return self

    def astype(self, dtype: Union[str, np.dtype]):
        """Change dtype of the object."""
        self.dtype = np.dtype(dtype)
        self.normalized_laplacian = self.normalized_laplacian.astype(self.dtype)
        self.weights_inv_sqrt = self.weights_inv_sqrt.astype(self.dtype)
        self.weights_inv_sqrt_vec = self.weights_inv_sqrt.astype(self.dtype)
        return self


class NormalizedAdjacencyOperator(LinearOperator):
    """Regularized normalized adjacency matrix as a Scipy LinearOperator.

    The normalized adjacency operator is then defined as
    :math:`\\bar{A} = D^{-1/2}AD^{-1/2}`.

    Parameters
    ----------
    adjacency :
        :term:`Adjacency <adjacency>` matrix of the graph.
    regularization : float
        Constant added to all entries of the adjacency matrix.

    Examples
    --------
    >>> from sknetwork.data import house
    >>> adjacency = house()
    >>> adj_norm = NormalizedAdjacencyOperator(adjacency, 0.)
    >>> x = np.sqrt(adjacency.dot(np.ones(5)))
    >>> np.allclose(x, adj_norm.dot(x))
    True
    """
    def __init__(self, adjacency: Union[sparse.csr_matrix, np.ndarray], regularization: float = 0.):
        super(NormalizedAdjacencyOperator, self).__init__(dtype=float, shape=adjacency.shape)
        self.adjacency = adjacency
        self.regularization = regularization

        n = self.adjacency.shape[0]
        self.weights_sqrt = np.sqrt(self.adjacency.dot(np.ones(n)) + self.regularization * n)

    def _matvec(self, matrix: np.ndarray):
        matrix = (matrix.T / self.weights_sqrt).T
        prod = self.adjacency.dot(matrix)
        if matrix.ndim == 2:
            prod += self.regularization * np.tile(matrix.sum(axis=0), (self.shape[0], 1))
        else:
            prod += self.regularization * matrix.sum()
        return (prod.T / self.weights_sqrt).T

    def _transpose(self):
        return self

    def astype(self, dtype: Union[str, np.dtype]):
        """Change dtype of the object."""
        self.dtype = np.dtype(dtype)
        self.adjacency = self.adjacency.astype(self.dtype)
        self.weights_sqrt = self.weights_sqrt.astype(self.dtype)

        return self


class CoNeighborOperator(LinearOperator):
    """Co-neighborhood adjacency as a LinearOperator.

    * Graphs
    * Digraphs
    * Bigraphs

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
    >>> coneigh = CoNeighborOperator(biadjacency)
    >>> np.allclose(d_out, coneigh.dot(np.ones(4)))
    True
    """
    def __init__(self, adjacency: Union[sparse.csr_matrix, np.ndarray], normalized: bool = True):
        adjacency = check_format(adjacency).astype(float)
        n = adjacency.shape[0]
        super(CoNeighborOperator, self).__init__(dtype=float, shape=(n, n))

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
        operator = CoNeighborOperator(self.backward)
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
