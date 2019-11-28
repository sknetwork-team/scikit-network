#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.basics.rand_walk import transition_matrix
from sknetwork.embedding.bispectral import BiSpectral
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
        Either ``'exact'`` or ``'knn'``. If 'exact' the output is computed with matrix multiplication.
        However, the density can be much higher than in the input graph and this can trigger Memory errors.
        If ``'knn'``, the co-neighborhood is approximated through KNN-search in an appropriate spectral embedding space.
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

    if method == 'exact':
        if normalized:
            forward = transition_matrix(adjacency.T)
        else:
            forward = adjacency.T
        return adjacency.dot(forward)

    elif method == 'knn':
        if normalized:
            bispectral = BiSpectral(embedding_dimension, weights='degree', col_weights='degree', scaling='divide')
        else:
            bispectral = BiSpectral(embedding_dimension, weights='degree', col_weights='uniform', scaling=None)

        bispectral.fit(adjacency)
        knn = KNeighborsTransformer(n_neighbors, undirected=True)
        knn.fit(bispectral.row_embedding_)
        return knn.adjacency_
    else:
        raise ValueError('method must be "exact" or "knn".')
