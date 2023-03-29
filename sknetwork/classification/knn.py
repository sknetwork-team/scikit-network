#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in November 2019
@author: Nathan de Lara <nathan.delara@polytechnique.org>
@author: Thomas Bonald <tbonald@enst.fr>
"""
from typing import Optional, Union

import numpy as np
from scipy import sparse

from sknetwork.classification.base import BaseClassifier
from sknetwork.embedding.base import BaseEmbedding
from sknetwork.linalg.normalization import get_norms, normalize
from sknetwork.utils.check import check_n_neighbors
from sknetwork.utils.format import get_adjacency_values


class KNN(BaseClassifier):
    """Node classification by K-nearest neighbors in the embedding space.

    For bipartite graphs, classify rows only (see ``BiKNN`` for joint classification of rows and columns).

    Parameters
    ----------
    embedding_method :
        Embedding method used to represent nodes in vector space.
        If ``None`` (default), use cosine similarity.
    n_neighbors :
        Number of nearest neighbors .

    Attributes
    ----------
    labels_ : np.ndarray, shape (n_labels,)
        Label of each node.
    membership_ : sparse.csr_matrix, shape (n_row, n_labels)
        Membership matrix.
    labels_row_ : np.ndarray
        Labels of rows, for bipartite graphs.
    labels_col_ : np.ndarray
        Labels of columns, for bipartite graphs.
    membership_row_ : sparse.csr_matrix, shape (n_row, n_labels)
        Membership matrix of rows, for bipartite graphs.
    membership_col_ : sparse.csr_matrix, shape (n_col, n_labels)
        Membership matrix of columns, for bipartite graphs.
    Example
    -------
    >>> from sknetwork.classification import KNN
    >>> from sknetwork.embedding import Spectral
    >>> from sknetwork.data import karate_club
    >>> knn = KNN(Spectral(2), n_neighbors=1)
    >>> graph = karate_club(metadata=True)
    >>> adjacency = graph.adjacency
    >>> labels_true = graph.labels
    >>> labels = {0: labels_true[0], 33: labels_true[33]}
    >>> labels_pred = knn.fit_predict(adjacency, labels)
    >>> np.round(np.mean(labels_pred == labels_true), 2)
    0.97
    """
    def __init__(self, embedding_method: Optional[BaseEmbedding] = None, n_neighbors: int = 3):
        super(KNN, self).__init__()
        self.embedding_method = embedding_method
        self.n_neighbors = n_neighbors
        self.bipartite = None

    @staticmethod
    def _instantiate_vars(labels: np.ndarray):
        index_seed = np.flatnonzero(labels >= 0)
        index_remain = np.flatnonzero(labels < 0)
        return index_seed, index_remain

    def _fit_core(self, embedding, labels, index_seed, index_remain):
        embedding_seed = embedding[index_seed]
        embedding_remain = embedding[index_remain]
        n_neighbors = check_n_neighbors(self.n_neighbors, len(index_seed))

        norms_seed = get_norms(embedding_seed, p=2)
        neighbors = []
        for i in range(len(index_remain)):
            vector = embedding_remain[i]
            if sparse.issparse(vector):
                vector = vector.toarray().ravel()
            distances = norms_seed**2 - 2 * embedding_seed.dot(vector) + np.sum(vector**2)
            neighbors += list(index_seed[np.argpartition(distances, n_neighbors)[:n_neighbors]])
        labels_neighbor = labels[neighbors]

        # membership matrix
        row = list(np.repeat(index_remain, n_neighbors))
        col = list(labels_neighbor)
        data = list(np.ones_like(labels_neighbor))

        row += list(index_seed)
        col += list(labels[index_seed])
        data += list(np.ones_like(index_seed))

        membership = normalize(sparse.csr_matrix((data, (row, col)), shape=(len(labels), np.max(labels) + 1)))
        labels = np.argmax(membership.toarray(), axis=1)

        return membership, labels

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray], labels: Union[np.ndarray, dict] = None,
            labels_row: Union[np.ndarray, dict] = None, labels_col: Union[np.ndarray, dict] = None) -> 'KNN':
        """Node classification by k-nearest neighbors in the embedding space.

        Parameters
        ----------
        input_matrix :
            Adjacency matrix or biadjacency matrix of the graph.
        labels :
            Known labels (dictionary or array). Negative values ignored.
        labels_row, labels_col :
            Labels of rows and columns (for bipartite graphs).

        Returns
        -------
        self: :class:`KNN`
        """
        adjacency, labels, self.bipartite = get_adjacency_values(input_matrix, values=labels, values_row=labels_row,
                                                                 values_col=labels_col)
        labels = labels.astype(int)
        index_seed, index_remain = self._instantiate_vars(labels)
        if self.embedding_method is None:
            embedding = normalize(adjacency, p=2)
        else:
            embedding = self.embedding_method.fit_transform(adjacency)
        membership, labels = self._fit_core(embedding, labels, index_seed, index_remain)

        self.membership_ = membership
        self.labels_ = labels

        if self.bipartite:
            self._split_vars(input_matrix.shape)

        return self
