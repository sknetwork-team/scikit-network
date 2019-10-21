#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

from typing import Union

from scipy import sparse
import numpy as np

from sknetwork.embedding.svd import SVD
from sknetwork.utils.checks import check_format
from sknetwork.utils.kneighbors import KNeighborsTransformer


def co_neighbors_graph(adjacency: Union[sparse.csr_matrix, np.ndarray], normalized: bool = True, method='knn',
                       n_neighbors: int = 5, embedding_dimension: int = 8) -> sparse.csr_matrix:
    """Compute the co-neighborhood adjacency defined as

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
        Either 'exact' or 'knn'. If 'exact' the output is computed with matrix multiplication.
        However, the density can be much higher than in the input graph and this can trigger Memory errors.
        If 'knn', the co-neighborhood is approximated through KNN-search in an appropriate spectral embedding space.
    n_neighbors:
        Number of neighbors for the KNN search. Only useful if ``method='knn'``.
    embedding_dimension:
        Dimension of the embedding space. Only useful if ``method='knn'``.

    Returns
    -------
    adjacency_: sparse.csr_matrix
        Adjacency of the co-neighborhood.

    """
    adjacency = check_format(adjacency)
    n1, n2 = adjacency.shape

    if method == 'exact':
        # pseudo inverse weight matrix
        if normalized:
            col_weights: np.ndarray = adjacency.T.dot(np.ones(n1))
            diag_feat = sparse.diags(col_weights, shape=(n2, n2), format='csr')
            diag_feat.data = 1 / diag_feat.data
        else:
            diag_feat = sparse.eye(n2, format='csr')
        return adjacency.dot(diag_feat).dot(adjacency.T)

    elif method == 'knn':
        if normalized:
            col_weights = 'degree'
        else:
            col_weights = 'uniform'
        gsvd = SVD(embedding_dimension, weights='uniform', col_weights=col_weights, scaling=None)
        gsvd.fit(adjacency)
        knn = KNeighborsTransformer(n_neighbors, make_undirected=True)
        knn.fit(gsvd.embedding_)
        return knn.adjacency_
    else:
        raise ValueError('method must be "exact" or "knn".')
