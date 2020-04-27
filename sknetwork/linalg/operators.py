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

from sknetwork.linalg.sparse_lowrank import SparseLR


class RegularizedAdjacency(SparseLR):
    """Regularized adjacency matrix as a Scipy LinearOperator.

    The regularized adjacency is formally defined as :math:`A_{\\alpha} = A + \\alpha 11^T`,
    or :math:`A_{\\alpha} = A + \\alpha d^{+}(d^{-})^T`
    where :math:`\\alpha` is the regularization parameter.

    Parameters
    ----------
    adjacency :
        :term:`Adjacency <adjacency>` matrix of the graph.
    regularization : float
        Constant implicitly added to all entries of the adjacency matrix.
    degree_mode : bool
        If `True`, the regularization parameter for entry (i, j) is scaled by out-degree of node i and
        in-degree of node j.
        If `False`, the regularization parameter is applied to all entries.

    Examples
    --------
    >>> from sknetwork.data import star_wars
    >>> biadjacency = star_wars(metadata=False)
    >>> biadj_reg = RegularizedAdjacency(biadjacency, 0.1, degree_mode=True)
    >>> biadj_reg.dot(np.ones(3))
    array([3.6, 1.8, 5.4, 3.6])
    """
    def __init__(self, adjacency: Union[sparse.csr_matrix, np.ndarray], regularization: float = 0.,
                 degree_mode: bool = False):
        n_row, n_col = adjacency.shape
        x = regularization * np.ones(n_row)
        y = np.ones(n_col)
        if degree_mode:
            x, y = adjacency.dot(y), adjacency.T.dot(x)
        super(RegularizedAdjacency, self).__init__(adjacency, (x, y))


class LaplacianOperator(LinearOperator):
    """Regularized Laplacian matrix as a Scipy LinearOperator.

    The Laplacian operator is then defined as :math:`L = D - A`.

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

    The normalized adjacency operator is then defined as
    :math:`\\bar{A} = D^{-1/2}AD^{-1/2}`.

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
