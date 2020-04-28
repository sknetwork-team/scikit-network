#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

from typing import Union

import numpy as np
from scipy import sparse

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
