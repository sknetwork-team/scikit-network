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
        Random number generator or random seed. If ``None``, numpy.random is used.
    isolated_nodes : str
        What to do with isolated column nodes. Can be ``'remove'`` (default), ``'merge'`` or ``'keep'``.

    Attributes
    ----------
    embedding_ : array, shape = (n, n_components)
        Embedding of the nodes.
    embedding_row_ : array, shape = (n_row, n_components)
        Embedding of the rows (copy of **embedding_**).
    embedding_col_ : array, shape = (n_col, n_components)
        Embedding of the columns.
    labels_row_ : np.ndarray
        Labels of the rows (used to build the embedding of the columns).
    labels_col_ : np.ndarray
        Labels of the columns (used to build the embedding of the rows).

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
    def __init__(self, resolution: float = 1, modularity: str = 'dugue', tol_optimization: float = 1e-3,
                 tol_aggregation: float = 1e-3, n_aggregations: int = -1, shuffle_nodes: bool = False,
                 random_state: Optional[Union[np.random.RandomState, int]] = None, isolated_nodes: str = 'remove'):
        super(BiLouvainEmbedding, self).__init__()
        self.resolution = np.float32(resolution)
        self.modularity = modularity.lower()
        self.tol_optimization = np.float32(tol_optimization)
        self.tol_aggregation = tol_aggregation
        self.n_aggregations = n_aggregations
        self.shuffle_nodes = shuffle_nodes
        self.random_state = check_random_state(random_state)
        self.isolated_nodes = isolated_nodes
        self.labels_row_ = None
        self.labels_col_ = None

    def fit(self, biadjacency: sparse.csr_matrix):
        """Embedding of bipartite graphs from the clustering obtained with Louvain.

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

        embedding_row = bilouvain.membership_row_
        embedding_col = bilouvain.membership_col_

        if self.isolated_nodes in ['remove', 'merge']:
            # remove or merge isolated column nodes and reindex labels
            labels_unique, counts = np.unique(bilouvain.labels_col_, return_counts=True)
            n_labels = max(labels_unique) + 1
            labels_old = labels_unique[counts > 1]
            if self.isolated_nodes == 'remove':
                labels_new = -np.ones(n_labels, dtype='int')
            else:
                labels_new = len(labels_old) * np.ones(n_labels, dtype='int')
            labels_new[labels_old] = np.arange(len(labels_old))
            labels_col = labels_new[bilouvain.labels_col_]

            # reindex row labels accordingly
            labels_unique = np.unique(bilouvain.labels_row_)
            n_labels = max(labels_unique) + 1
            labels_new = -np.ones(n_labels, dtype='int')
            labels_new[labels_old] = np.arange(len(labels_old))
            labels_row = labels_new[bilouvain.labels_row_]

            # get embeddings
            probs = normalize(biadjacency)
            embedding_row = probs.dot(membership_matrix(labels_col))
            probs = normalize(biadjacency.T)
            embedding_col = probs.dot(membership_matrix(labels_row))

        self.labels_row_ = bilouvain.labels_row_
        self.labels_col_ = bilouvain.labels_col_

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
        n = self.labels_col_.shape[0]

        adjacency_vectors = check_adjacency_vector(adjacency_vectors, n)
        check_nonnegative(adjacency_vectors)
        membership = membership_matrix(self.labels_col_)

        return normalize(adjacency_vectors).dot(membership)


class LouvainEmbedding(BaseEmbedding):
    """Embedding of graphs induced by Louvain clustering. Each component of the embedding corresponds
    to a cluster obtained by Louvain.

    Parameters
    ----------
    resolution : float
        Resolution parameter.
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
    isolated_nodes : str
        What to do with isolated nodes. Can be ``'remove'`` (default), ``'merge'`` or ``'keep'``.

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
    def __init__(self, resolution: float = 1, modularity: str = 'dugue',  tol_optimization: float = 1e-3,
                 tol_aggregation: float = 1e-3, n_aggregations: int = -1, shuffle_nodes: bool = False,
                 random_state: Optional[Union[np.random.RandomState, int]] = None, isolated_nodes: str = 'remove'):
        super(LouvainEmbedding, self).__init__()
        self.resolution = np.float32(resolution)
        self.modularity = modularity.lower()
        self.tol_optimization = np.float32(tol_optimization)
        self.tol_aggregation = tol_aggregation
        self.n_aggregations = n_aggregations
        self.shuffle_nodes = shuffle_nodes
        self.random_state = check_random_state(random_state)
        self.isolated_nodes = isolated_nodes

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

        if self.isolated_nodes in ['remove', 'merge']:
            # remove or merge isolated nodes and reindex labels
            labels_unique, counts = np.unique(louvain.labels_, return_counts=True)
            n_labels = max(labels_unique) + 1
            labels_old = labels_unique[counts > 1]
            if self.isolated_nodes == 'remove':
                labels_new = -np.ones(n_labels, dtype='int')
            else:
                labels_new = len(labels_old) * np.ones(n_labels, dtype='int')
            labels_new[labels_old] = np.arange(len(labels_old))
            labels_ = labels_new[louvain.labels_]

            # get embeddings
            probs = normalize(adjacency)
            embedding_ = probs.dot(membership_matrix(labels_))

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

        return normalize(adjacency_vectors.dot(membership))
