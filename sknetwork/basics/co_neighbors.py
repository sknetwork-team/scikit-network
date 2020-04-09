#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

from typing import Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from sknetwork.embedding.spectral import BiSpectral
from sknetwork.linalg.normalization import normalize
from sknetwork.utils.check import check_format
from sknetwork.utils.knn import KNNDense


def co_neighbors_graph(adjacency: Union[sparse.csr_matrix, np.ndarray], normalized: bool = True, method='knn',
                       n_neighbors: int = 5, n_components: int = 8) -> sparse.csr_matrix:
    """Compute the co-neighborhood adjacency.

    * Graphs
    * Digraphs
    * Bigraphs

    :math:`\\tilde{A} = AF^{-1}A^T`,

    where F is a weight matrix.

    Parameters
    ----------
    adjacency:
        Adjacency of the input graph.
    normalized:
        If ``True``, F is the diagonal in-degree matrix :math:`F = \\text{diag}(A^T1)`.
        Otherwise, F is the identity matrix.
    method:
        Either ``'exact'`` or ``'knn'``. If 'exact' the output is computed with matrix multiplication.
        However, the density can be much higher than in the input graph and this can trigger Memory errors.
        If ``'knn'``, the co-neighborhood is approximated through KNNDense-search in an appropriate spectral embedding
        space.
    n_neighbors:
        Number of neighbors for the KNNDense search. Only useful if ``method='knn'``.
    n_components:
        Dimension of the embedding space. Only useful if ``method='knn'``.

    Returns
    -------
    adjacency : sparse.csr_matrix
        Adjacency of the co-neighborhood.

    """
    adjacency = check_format(adjacency)

    if method == 'exact':
        if normalized:
            forward = normalize(adjacency.T).tocsr()
        else:
            forward = adjacency.T
        return adjacency.dot(forward)

    elif method == 'knn':
        bispectral = BiSpectral(n_components, normalized_laplacian=normalized)
        bispectral.fit(adjacency)
        knn = KNNDense(n_neighbors, undirected=True)
        knn.fit(bispectral.embedding_row_)
        return knn.adjacency_
    else:
        raise ValueError('method must be "exact" or "knn".')


class CoNeighbors(LinearOperator):
    """Co-neighborhood adjacency as a LinearOperator.

    * Graphs
    * Digraphs
    * Bigraphs

    :math:`\\tilde{A} = AF^{-1}A^T`,

    where F is a weight matrix.

    Parameters
    ----------
    adjacency:
        Adjacency of the input graph.
    normalized:
        If ``True``, F is the diagonal in-degree matrix :math:`F = \\text{diag}(A^T1)`.
        Otherwise, F is the identity matrix.

    Returns
    -------
    LinearOperator
        Adjacency of the co-neighborhood.
    """
    def __init__(self, adjacency: Union[sparse.csr_matrix, np.ndarray], normalized: bool = True):
        adjacency = check_format(adjacency)
        n = adjacency.shape[0]
        super(CoNeighbors, self).__init__(dtype=float, shape=(n, n))

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
        """Transposed matrix.

        Returns
        -------
        CoNeighbors object
        """
        operator = CoNeighbors(self.backward)
        operator.backward = self.forward.T.tocsr()
        operator.forward = self.backward.T.tocsr()
        return operator

    def _adjoint(self):
        return self.transpose()

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
