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


class LaplacianOperator(LinearOperator):
    """Regularized Laplacian matrix as a Scipy LinearOperator.

    The regularized adjacency is formally defined as :math:`A_{\\alpha} = A + \\alpha 11^T`,
    where :math:`\\alpha` is the regularization parameter.

    The Laplacian operator is then defined as :math:`L = D_{\\alpha} - A_{\\alpha}`.

    Parameters
    ----------
    adjacency :
        :term:`Adjacency <adjacency>` matrix of the graph.
    regularization : float
        Constant implicitly added to all entries of the adjacency matrix.

    Examples
    --------
    >>> from sknetwork.data import house
    >>> adjacency = house()
    >>> laplacian = LaplacianOperator(adjacency, 0.1)
    >>> laplacian.dot(np.ones(adjacency.shape[1]))
    array([0., 0., 0., 0., 0.])
    """
    def __init__(self, adjacency: Union[sparse.csr_matrix, np.ndarray], regularization: float = 0.):
        super(LaplacianOperator, self).__init__(dtype=float, shape=adjacency.shape)
        self.regularization = regularization
        self.weights = adjacency.dot(np.ones(adjacency.shape[1]))
        self.laplacian = sparse.diags(self.weights, format='csr') - adjacency

    def _matvec(self, matrix: np.ndarray):
        prod = self.laplacian.dot(matrix)
        prod += self.shape[0] * self.regularization * matrix
        if len(matrix.shape) == 2:
            prod -= self.regularization * np.tile(matrix.sum(axis=0), (self.shape[0], 1))
        else:
            prod -= self.regularization * matrix.sum()

        return prod

    def _transpose(self):
        return self

    def astype(self, dtype: Union[str, np.dtype]):
        """Change dtype of the object."""
        self.dtype = np.dtype(dtype)
        self.laplacian = self.laplacian.astype(self.dtype)
        self.weights = self.weights.astype(self.dtype)

        return self


class NormalizedAdjacencyOperator(LinearOperator):
    """Regularized normalized adjacency matrix as a Scipy LinearOperator.

    The regularized adjacency is formally defined as :math:`A_{\\alpha} = A + \\alpha 11^T`,
    where :math:`\\alpha` is the regularization parameter.

    The normalized adjacency operator is then defined as
    :math:`\\bar{A} = D_{\\alpha}^{-1/2}A_{\\alpha}D_{\\alpha}^{-1/2}`.

    Parameters
    ----------
    adjacency :
        :term:`Adjacency <adjacency>` matrix of the graph.
    regularization : float
        Constant implicitly added to all entries of the adjacency matrix.

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
        if len(matrix.shape) == 2:
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
