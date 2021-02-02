#!/usr/bin/env python3
# coding: utf-8
"""
Created on Sep 2020
@author: Quentin Lutz <qlutz@enst.fr>
"""
from typing import Optional, Union

import numpy as np
from scipy import sparse

from sknetwork.clustering.louvain import BiLouvain, Louvain
from sknetwork.embedding.base import BaseBiEmbedding, BaseEmbedding
from sknetwork.linalg.normalization import normalize
from sknetwork.utils.check import check_random_state, check_adjacency_vector, check_nonnegative
from sknetwork.utils.membership import membership_matrix


class BiLouvainEmbedding(BaseBiEmbedding):
    """Embedding of bipartite graphs induced by Louvain clustering. Each component of the embedding corresponds
    to a cluster obtained by Louvain.

    Parameters
    ----------
    resolution : float
        Resolution parameter.
    remove_isolated : bool
        Denotes if clusters consisting of just one node should be removed.
    merge_isolated : bool
        Denotes if clusters consisting of just one node should be merged.
    modularity : str
        Which objective function to maximize. Can be ``'dugue'``, ``'newman'`` or ``'potts'``.
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
        Random number generator or random seed. If None, numpy.random is used.

    Attributes
    ----------
    embedding_ : array, shape = (n, n_components)
        Embedding of the nodes.
    embedding_row_ : array, shape = (n_row, n_components)
        Embedding of the rows (copy of **embedding_**).
    embedding_col_ : array, shape = (n_col, n_components)
        Embedding of the columns.

    Example
    -------
    >>> from sknetwork.embedding import BiLouvainEmbedding
    >>> from sknetwork.data import movie_actor
    >>> bilouvain = BiLouvainEmbedding()
    >>> biadjacency = movie_actor()
    >>> embedding = bilouvain.fit_transform(biadjacency)
    >>> embedding.shape
    (15, 4)
    """
    def __init__(self, resolution: float = 1, remove_isolated: bool = True,  merge_isolated: bool = False,
                 modularity: str = 'dugue', tol_optimization: float = 1e-3, tol_aggregation: float = 1e-3,
                 n_aggregations: int = -1, shuffle_nodes: bool = False,
                 random_state: Optional[Union[np.random.RandomState, int]] = None):
        super(BiLouvainEmbedding, self).__init__()
        self.resolution = np.float32(resolution)
        self.modularity = modularity.lower()
        self.tol_optimization = np.float32(tol_optimization)
        self.tol_aggregation = tol_aggregation
        self.n_aggregations = n_aggregations
        self.shuffle_nodes = shuffle_nodes
        self.random_state = check_random_state(random_state)
        self.merge_isolated = merge_isolated
        self.remove_isolated = remove_isolated

        self.labels_ = None

    def fit(self, biadjacency: sparse.csr_matrix):
        """Embedding of bipartite graphs from a clustering obtained with Louvain.

        Parameters
        ----------
        biadjacency:
            Biadjacency matrix of the graph.

        Returns
        -------
        self: :class:`BiLouvainEmbedding`
        """
        bilouvain = BiLouvain(resolution=self.resolution, modularity=self.modularity,
                              tol_optimization=self.tol_optimization, tol_aggregation=self.tol_aggregation,
                              n_aggregations=self.n_aggregations, shuffle_nodes=self.shuffle_nodes, sort_clusters=False,
                              return_membership=True, return_aggregate=True, random_state=self.random_state)
        bilouvain.fit(biadjacency)

        self.labels_ = bilouvain.labels_

        embedding_row = bilouvain.membership_row_
        embedding_col = bilouvain.membership_col_

        if self.remove_isolated:
            labels_row = bilouvain.labels_row_
            labels_col = bilouvain.labels_col_

            # remove singletons from column labels
            labels_unique, counts = np.unique(labels_col, return_counts=True)
            labels_new = -np.ones(max(labels_unique) + 1, dtype='int')
            labels_old = labels_unique[counts > 1]
            labels_new[labels_old] = np.arange(len(labels_old))
            labels_col = labels_new[labels_col]

            # reindex row labels accordingly
            labels_unique = np.unique(labels_row)
            labels_new = -np.ones(max(labels_unique) + 1, dtype='int')
            labels_new[labels_old] = np.arange(len(labels_old))
            labels_row = labels_new[labels_row]

            # embedding
            probs = normalize(biadjacency)
            embedding_row = probs.dot(membership_matrix(labels_col))
            probs = normalize(biadjacency.T)
            embedding_col = probs.dot(membership_matrix(labels_row))

        if self.merge_isolated:
            _, counts_row = np.unique(bilouvain.labels_row_, return_counts=True)
            n_isolated_nodes_row = (counts_row == 1).sum()
            if n_isolated_nodes_row:
                size_row = (biadjacency.shape[0], len(counts_row))
                embedding_row.resize(size_row)
                labels_row = bilouvain.labels_row_
                labels_row[-n_isolated_nodes_row:] = labels_row[-n_isolated_nodes_row]
                merge_labels_row = np.arange(len(counts_row), dtype=int)
                merge_labels_row[-n_isolated_nodes_row:] = merge_labels_row[-n_isolated_nodes_row]
                combiner_row = membership_matrix(merge_labels_row)
                embedding_row = embedding_row.dot(combiner_row)
                self.labels_ = labels_row

            _, counts_col = np.unique(bilouvain.labels_col_, return_counts=True)
            n_isolated_nodes_col = (counts_col == 1).sum()
            if n_isolated_nodes_col:
                size_col = (biadjacency.shape[1], len(counts_col))
                embedding_col.resize(size_col)
                merge_labels_col = np.arange(embedding_col.shape[1], dtype=int)
                merge_labels_col[-n_isolated_nodes_col:] = merge_labels_col[-n_isolated_nodes_col]
                combiner_col = membership_matrix(merge_labels_col)
                embedding_col = embedding_col.dot(combiner_col)

        self.embedding_row_ = embedding_row.toarray()
        self.embedding_col_ = embedding_col.toarray()
        self.embedding_ = self.embedding_row_

        return self

    def predict(self, adjacency_vectors: Union[sparse.csr_matrix, np.ndarray]) -> np.ndarray:
        """Predict the embedding of new rows, defined by their adjacency vectors.

        Parameters
        ----------
        adjacency_vectors :
            Adjacency vectors of rows.
            Array of shape (n_col,) (single vector) or (n_vectors, n_col)

        Returns
        -------
        embedding_vectors : np.ndarray
            Embedding of the nodes.
        """
        self._check_fitted()
        n = self.embedding_.shape[0]

        adjacency_vectors = check_adjacency_vector(adjacency_vectors, n)
        check_nonnegative(adjacency_vectors)
        membership = membership_matrix(self.labels_)

        return adjacency_vectors.dot(membership)


class LouvainEmbedding(BaseEmbedding):
    """Embedding of graphs induced by Louvain clustering. Each component of the embedding corresponds
    to a cluster obtained by Louvain.

    Parameters
    ----------
    resolution : float
        Resolution parameter.
    remove_isolated : bool
        Denotes if clusters consisting of just one node should be removed.
    merge_isolated : bool
        Denotes if clusters consisting of just one node should be merged.
    modularity : str
        Which objective function to maximize. Can be ``'dugue'``, ``'newman'`` or ``'potts'``.
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
        Random number generator or random seed. If None, numpy.random is used.

    Attributes
    ----------
    embedding_ : array, shape = (n, n_components)
        Embedding of the nodes.

    Example
    -------
    >>> from sknetwork.embedding import LouvainEmbedding
    >>> from sknetwork.data import karate_club
    >>> louvain = LouvainEmbedding()
    >>> adjacency = karate_club()
    >>> embedding = louvain.fit_transform(adjacency)
    >>> embedding.shape
    (34, 4)
    """
    def __init__(self, resolution: float = 1, remove_isolated: bool = True,  merge_isolated: bool = False,
                 modularity: str = 'dugue', tol_optimization: float = 1e-3, tol_aggregation: float = 1e-3,
                 n_aggregations: int = -1, shuffle_nodes: bool = False,
                 random_state: Optional[Union[np.random.RandomState, int]] = None):
        super(LouvainEmbedding, self).__init__()
        self.resolution = np.float32(resolution)
        self.modularity = modularity.lower()
        self.tol_optimization = np.float32(tol_optimization)
        self.tol_aggregation = tol_aggregation
        self.n_aggregations = n_aggregations
        self.shuffle_nodes = shuffle_nodes
        self.random_state = check_random_state(random_state)
        self.merge_isolated = merge_isolated
        self.remove_isolated = remove_isolated

        self.labels_ = None

    def fit(self, adjacency: sparse.csr_matrix):
        """Embedding of bipartite graphs from a clustering obtained with Louvain.

        Parameters
        ----------
        adjacency:
            Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`BiLouvainEmbedding`
        """
        louvain = Louvain(resolution=self.resolution, modularity=self.modularity,
                          tol_optimization=self.tol_optimization, tol_aggregation=self.tol_aggregation,
                          n_aggregations=self.n_aggregations, shuffle_nodes=self.shuffle_nodes, sort_clusters=True,
                          return_membership=True, return_aggregate=True, random_state=self.random_state)
        louvain.fit(adjacency)

        self.labels_ = louvain.labels_

        embedding_ = louvain.membership_

        if self.remove_isolated:
            _, counts = np.unique(louvain.labels_, return_counts=True)
            non_isolated_nodes = (counts > 1)
            embedding_ = embedding_[:, non_isolated_nodes]

        if self.merge_isolated:
            _, counts = np.unique(louvain.labels_, return_counts=True)
            n_isolated_nodes = (counts == 1).sum()
            if n_isolated_nodes:
                size = (adjacency.shape[0], len(counts))
                embedding_.resize(size)
                labels = louvain.labels_
                labels[-n_isolated_nodes:] = labels[-n_isolated_nodes]
                merge_labels = np.arange(len(counts), dtype=int)
                merge_labels[-n_isolated_nodes:] = merge_labels[-n_isolated_nodes]
                combiner = membership_matrix(merge_labels)
                embedding_ = embedding_.dot(combiner)
                self.labels_ = labels

        self.embedding_ = embedding_.toarray()

        return self

    def predict(self, adjacency_vectors: Union[sparse.csr_matrix, np.ndarray]) -> np.ndarray:
        """Predict the embedding of new rows, defined by their adjacency vectors.

        Parameters
        ----------
        adjacency_vectors :
            Adjacency vectors of rows.
            Array of shape (n_col,) (single vector) or (n_vectors, n_col)

        Returns
        -------
        embedding_vectors : np.ndarray
            Embedding of the nodes.
        """
        self._check_fitted()
        n = self.embedding_.shape[0]

        adjacency_vectors = check_adjacency_vector(adjacency_vectors, n)
        check_nonnegative(adjacency_vectors)
        membership = membership_matrix(self.labels_)

        return adjacency_vectors.dot(membership)
