#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in March 2023
@author: Thomas Bonald <thomas.bonald@telecom-paris.fr>
"""
from typing import Optional, Union

import numpy as np
from scipy import sparse

from sknetwork.linkpred.base import BaseLinker
from sknetwork.embedding.base import BaseEmbedding
from sknetwork.linalg.normalizer import normalize
from sknetwork.utils.check import check_n_neighbors
from sknetwork.utils.format import get_adjacency


class NNLinker(BaseLinker):
    """Link prediction by nearest neighbors in the embedding space, using cosine similarity.

    For bipartite graphs, predict links between rows and columns only.

    Parameters
    ----------
    n_neighbors : int
        Number of nearest neighbors. If ``None``, all nodes are considered.
    threshold : float
        Threshold on cosine similarity. Only links above this threshold are kept.
    embedding_method : :class:`BaseEmbedding`
        Embedding method used to represent nodes in vector space.
        If ``None`` (default), use identity.

    Attributes
    ----------
    links_ : sparse.csr_matrix
        Link matrix.

    Example
    -------
    >>> from sknetwork.linkpred import NNLinker
    >>> from sknetwork.data import karate_club
    >>> linker = NNLinker(n_neighbors=5, threshold=0.5)
    >>> graph = karate_club(metadata=True)
    >>> adjacency = graph.adjacency
    >>> links = linker.fit_predict(adjacency)
    >>> links.shape
    (34, 34)
    """
    def __init__(self, n_neighbors: Optional[int] = 10, threshold: float = 0,
                 embedding_method: Optional[BaseEmbedding] = None):
        super(NNLinker, self).__init__()
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        self.embedding_method = embedding_method
        self.bipartite = None

    def _fit_core(self, embedding, mask):
        n = embedding.shape[0]
        n_row = len(mask)
        if n_row < n:
            # bipartite graphs
            index_col = np.arange(n_row, n)
            n_col = n - n_row
        else:
            index_col = np.arange(n)
            n_col = n
        n_neighbors = check_n_neighbors(self.n_neighbors, len(index_col))

        row = []
        col = []
        data = []

        for i in np.flatnonzero(mask):
            vector = embedding[i]
            if sparse.issparse(vector):
                vector = vector.toarray().ravel()
            similarities = embedding[index_col].dot(vector)
            nn = np.argpartition(-similarities, n_neighbors)[:n_neighbors]
            mask_nn = np.zeros(n_col, dtype=bool)
            mask_nn[nn] = 1
            mask_nn[similarities < self.threshold] = 0
            nn = np.flatnonzero(mask_nn)

            row += len(nn) * [i]
            col += list(nn)
            data += list(similarities[nn])

        links = sparse.csr_matrix((data, (row, col)), shape=(n_row, n_col))

        return links

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray], index: Optional[np.ndarray] = None) -> 'NNLinker':
        """Link prediction by nearest neighbors in the embedding space, using cosine similarity

        Parameters
        ----------
        input_matrix : sparse.csr_matrix, np.ndarray
            Adjacency matrix or biadjacency matrix of the graph.
        index : np.ndarray
            Index of source nodes to consider. If ``None``, the links are predicted for all nodes.

        Returns
        -------
        self: :class:`NN`
        """
        n_row, _ = input_matrix.shape

        adjacency, self.bipartite = get_adjacency(input_matrix)

        if index is None:
            index = np.arange(n_row)
        mask = np.zeros(n_row, dtype=bool)
        mask[index] = 1

        if self.embedding_method is None:
            embedding = adjacency
        else:
            embedding = self.embedding_method.fit_transform(adjacency)

        embedding = normalize(embedding, p=2)
        links = self._fit_core(embedding, mask)

        self.links_ = links

        return self
