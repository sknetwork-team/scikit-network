#!/usr/bin/env python3
# coding: utf-8
"""
Created in September 2020
@author: Quentin Lutz <qlutz@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""
from typing import Optional, Union

import numpy as np
from scipy import sparse

from sknetwork.clustering.louvain import Louvain
from sknetwork.embedding.base import BaseEmbedding
from sknetwork.linalg.normalizer import normalize
from sknetwork.utils.check import check_random_state, check_adjacency_vector, check_nonnegative, is_square
from sknetwork.utils.membership import get_membership


def reindex_labels(labels: np.ndarray, labels_secondary: Optional[np.ndarray] = None, which: str = 'remove'):
    """Reindex labels, removing or merging labels of count 1."""
    labels_unique, counts = np.unique(labels, return_counts=True)
    n_labels = max(labels_unique) + 1
    labels_keep = labels_unique[counts > 1]
    if which == 'remove':
        label_index = -np.ones(n_labels, dtype='int')
        label_index[labels_keep] = np.arange(len(labels_keep))
    elif which == 'merge':
        label_index = len(labels_keep) * np.ones(n_labels, dtype='int')
        label_index[labels_keep] = np.arange(len(labels_keep))
    else:
        label_index = np.arange(n_labels)
    labels = label_index[labels]
    if labels_secondary is not None:
        labels_unique = np.unique(labels_secondary)
        n_labels = max(labels_unique) + 1
        label_index = -np.ones(n_labels, dtype='int')
        label_index[labels_keep] = np.arange(len(labels_keep))
        labels_secondary = label_index[labels_secondary]
    return labels, labels_secondary


class LouvainEmbedding(BaseEmbedding):
    """Embedding of graphs induced by Louvain clustering. Each component of the embedding corresponds
    to a cluster obtained by Louvain.

    Parameters
    ----------
    resolution : float
        Resolution parameter.
    modularity : str
        Which objective function to maximize. Can be ``'Dugue'``, ``'Newman'`` or ``'Potts'``.
    tol_optimization :
        Minimum increase in the objective function to enter a new optimization pass.
    tol_aggregation :
        Minimum increase in the objective function to enter a new aggregation pass.
    n_aggregations :
        Maximum number of aggregations.
        A negative value is interpreted as no limit.
    shuffle_nodes :
        Enables node shuffling before optimization.
    random_state :
        Random number generator or random seed. If ``None``, numpy.random is used.
    isolated_nodes : str
        What to do with isolated column nodes. Can be ``'remove'`` (default), ``'merge'`` or ``'keep'``.

    Attributes
    ----------
    embedding_ : array, shape = (n, n_components)
        Embedding of the nodes.
    embedding_row_ : array, shape = (n_row, n_components)
        Embedding of the rows, for bipartite graphs.
    embedding_col_ : array, shape = (n_col, n_components)
        Embedding of the columns, for bipartite graphs.
    labels_row_ : np.ndarray
        Labels of the rows (used to build the embedding of the columns).
    labels_col_ : np.ndarray
        Labels of the columns (used to build the embedding of the rows).

    Example
    -------
    >>> from sknetwork.embedding import LouvainEmbedding
    >>> from sknetwork.data import house
    >>> louvain = LouvainEmbedding()
    >>> adjacency = house()
    >>> embedding = louvain.fit_transform(adjacency)
    >>> embedding.shape
    (5, 2)
    """
    def __init__(self, resolution: float = 1, modularity: str = 'Dugue', tol_optimization: float = 1e-3,
                 tol_aggregation: float = 1e-3, n_aggregations: int = -1, shuffle_nodes: bool = False,
                 random_state: Optional[Union[np.random.RandomState, int]] = None, isolated_nodes: str = 'remove'):
        super(LouvainEmbedding, self).__init__()
        self.resolution = resolution
        self.modularity = modularity.lower()
        self.tol_optimization = tol_optimization
        self.tol_aggregation = tol_aggregation
        self.n_aggregations = n_aggregations
        self.shuffle_nodes = shuffle_nodes
        self.random_state = check_random_state(random_state)
        self.isolated_nodes = isolated_nodes

        self.labels_ = None
        self.embedding_ = None
        self.embedding_row_ = None
        self.embedding_col_ = None

    def fit(self, input_matrix: sparse.csr_matrix, force_bipartite: bool = False):
        """Embedding of graphs from the clustering obtained with Louvain.

        Parameters
        ----------
        input_matrix :
            Adjacency matrix or biadjacency matrix of the graph.
        force_bipartite : bool (default = ``False``)
            If ``True``, force the input matrix to be considered as a biadjacency matrix.
        Returns
        -------
        self: :class:`BiLouvainEmbedding`
        """
        louvain = Louvain(resolution=self.resolution, modularity=self.modularity,
                          tol_optimization=self.tol_optimization, tol_aggregation=self.tol_aggregation,
                          n_aggregations=self.n_aggregations, shuffle_nodes=self.shuffle_nodes, sort_clusters=False,
                          return_probs=True, return_aggregate=True, random_state=self.random_state)
        louvain.fit(input_matrix, force_bipartite=force_bipartite)

        # isolated nodes
        if is_square(input_matrix):
            labels = louvain.labels_
            labels_secondary = None
        else:
            labels = louvain.labels_col_
            labels_secondary = louvain.labels_row_

        self.labels_, labels_row = reindex_labels(labels, labels_secondary, self.isolated_nodes)

        # embedding
        probs = normalize(input_matrix)
        embedding_ = probs.dot(get_membership(self.labels_))
        self.embedding_ = embedding_.toarray()

        if labels_row is not None:
            probs = normalize(input_matrix.T)
            embedding_col = probs.dot(get_membership(labels_row))
            self.embedding_row_ = self.embedding_
            self.embedding_col_ = embedding_col.toarray()

        return self

    def predict(self, adjacency_vectors: Union[sparse.csr_matrix, np.ndarray]) -> np.ndarray:
        """Predict the embedding of new rows, defined by their adjacency vectors.

        Parameters
        ----------
        adjacency_vectors :
            Adjacency row vectors.
            Array of shape (n_col,) (single vector) or (n_vectors, n_col)

        Returns
        -------
        embedding_vectors : np.ndarray
            Embedding of the nodes.
        """
        self._check_fitted()
        if self.embedding_col_ is not None:
            n = len(self.embedding_col_)
        else:
            n = len(self.embedding_)

        adjacency_vectors = check_adjacency_vector(adjacency_vectors, n)
        check_nonnegative(adjacency_vectors)
        membership = get_membership(self.labels_)

        return normalize(adjacency_vectors).dot(membership)
