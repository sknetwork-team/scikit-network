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
from sknetwork.linalg.normalizer import get_norms, normalize
from sknetwork.utils.check import check_n_neighbors
from sknetwork.utils.format import get_adjacency_values


class NNClassifier(BaseClassifier):
    """Node classification by K-nearest neighbors in the embedding space.

    Parameters
    ----------
    n_neighbors : int
        Number of nearest neighbors .
    embedding_method : :class:`BaseEmbedding`
        Embedding method used to represent nodes in vector space.
        If ``None`` (default), use identity.
    normalize : bool
        If ``True``, apply normalization so that all vectors have norm 1 in the embedding space.

    Attributes
    ----------
    labels_ : np.ndarray, shape (n_labels,)
        Labels of nodes.
    probs_ : sparse.csr_matrix, shape (n_row, n_labels)
        Probability distribution over labels.
    labels_row_ : np.ndarray
        Labels of rows, for bipartite graphs.
    labels_col_ : np.ndarray
        Labels of columns, for bipartite graphs.
    probs_row_ : sparse.csr_matrix, shape (n_row, n_labels)
        Probability distributions over labels of rows, for bipartite graphs.
    probs_col_ : sparse.csr_matrix, shape (n_col, n_labels)
        Probability distributions over labels of columns, for bipartite graphs.

    Example
    -------
    >>> from sknetwork.classification import NNClassifier
    >>> from sknetwork.data import karate_club
    >>> classifier = NNClassifier(n_neighbors=1)
    >>> graph = karate_club(metadata=True)
    >>> adjacency = graph.adjacency
    >>> labels_true = graph.labels
    >>> labels = {0: labels_true[0], 33: labels_true[33]}
    >>> labels_pred = classifier.fit_predict(adjacency, labels)
    >>> float(round(np.mean(labels_pred == labels_true), 2))
    0.82
    """
    def __init__(self, n_neighbors: int = 3, embedding_method: Optional[BaseEmbedding] = None, normalize: bool = True):
        super(NNClassifier, self).__init__()
        self.n_neighbors = n_neighbors
        self.embedding_method = embedding_method
        self.normalize = normalize

    @staticmethod
    def _instantiate_vars(labels: np.ndarray):
        index_train = np.flatnonzero(labels >= 0)
        index_test = np.flatnonzero(labels < 0)
        return index_train, index_test

    def _fit_core(self, embedding, labels, index_train, index_test):
        n_neighbors = check_n_neighbors(self.n_neighbors, len(index_train))

        norms_train = get_norms(embedding[index_train], p=2)
        neighbors = []
        for i in index_test:
            vector = embedding[i]
            if sparse.issparse(vector):
                vector = vector.toarray().ravel()
            distances = norms_train**2 - 2 * embedding[index_train].dot(vector) + np.sum(vector**2)
            neighbors += list(index_train[np.argpartition(distances, n_neighbors)[:n_neighbors]])
        labels_neighbor = labels[neighbors]

        # membership matrix
        row = list(np.repeat(index_test, n_neighbors))
        col = list(labels_neighbor)
        data = list(np.ones_like(labels_neighbor))

        row += list(index_train)
        col += list(labels[index_train])
        data += list(np.ones_like(index_train))

        probs = normalize(sparse.csr_matrix((data, (row, col)), shape=(len(labels), np.max(labels) + 1)))
        labels = np.argmax(probs.toarray(), axis=1)

        return probs, labels

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray], labels: Union[np.ndarray, list, dict] = None,
            labels_row: Union[np.ndarray, list, dict] = None,
            labels_col: Union[np.ndarray, list, dict] = None) -> 'NNClassifier':
        """Node classification by k-nearest neighbors in the embedding space.

        Parameters
        ----------
        input_matrix : sparse.csr_matrix, np.ndarray
            Adjacency matrix or biadjacency matrix of the graph.
        labels : np.ndarray, dict
            Known labels. Negative values ignored.
        labels_row : np.ndarray, dict
            Known labels of rows, for bipartite graphs.
        labels_col : np.ndarray, dict
            Known labels of columns, for bipartite graphs.

        Returns
        -------
        self: :class:`KNN`
        """
        adjacency, labels, self.bipartite = get_adjacency_values(input_matrix, values=labels, values_row=labels_row,
                                                                 values_col=labels_col)
        labels = labels.astype(int)
        index_seed, index_remain = self._instantiate_vars(labels)

        if self.embedding_method is None:
            embedding = adjacency
        else:
            embedding = self.embedding_method.fit_transform(adjacency)

        if self.normalize:
            embedding = normalize(embedding, p=2)

        probs, labels = self._fit_core(embedding, labels, index_seed, index_remain)

        self.labels_ = labels
        self.probs_ = probs
        self._split_vars(input_matrix.shape)

        return self
