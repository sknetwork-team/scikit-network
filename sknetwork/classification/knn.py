#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov, 2019
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <tbonald@enst.fr>
"""

import warnings

from typing import Optional, Union

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from sknetwork.classification import BaseClassifier
from sknetwork.embedding import BaseEmbedding, GSVD
from sknetwork.linalg.normalization import normalize
from sknetwork.utils.check import check_seeds


class KNN(BaseClassifier):
    """Node classification by K-nearest neighbors in the embedding space.

    Graphs, digraphs and bigraphs.

    For bigraphs, classify rows only (see ``BiKNN``for joint classification of rows and columns).

    Parameters
    ----------
    embedding_method :
        Which algorithm to use to project the nodes in vector space. Default is ``GSVD``.
    n_neighbors :
        Number of nearest neighbors to consider.
    factor_distance :
        Power weighting factor :math:``\\alpha`` applied to the distance to each neighbor.
        Neighbor at distance :math:``d`` has weight :math:``1 / d^\\alpha``. Default is 2.
    leaf_size :
        Leaf size passed to KDTree.
    p :
        Which Minkowski p-norm to use. Default is 2 (Euclidean distance).
    tol_nn :
        Tolerance in nearest neighbors search; the k-th returned value is guaranteed to be no further
        than ``1 + tol_nn`` times the distance to the actual k-th nearest neighbor.
    n_jobs :
        Number of jobs to schedule for parallel processing. If -1 is given all processors are used.

    Attributes
    ----------
    labels_ : np.ndarray
        Label of each node.
    membership_ : sparse.csr_matrix
        Membership matrix (columns = labels).

    Example
    -------
    >>> from sknetwork.data import karate_club
    >>> from sknetwork.embedding import GSVD
    >>> knn = KNN(GSVD(3), n_neighbors=1)
    >>> graph = karate_club(metadata=True)
    >>> adjacency = graph.adjacency
    >>> labels_true = graph.labels
    >>> seeds = {0: labels_true[0], 33: labels_true[33]}
    >>> labels_pred = knn.fit_transform(adjacency, seeds)
    >>> np.round(np.mean(labels_pred == labels_true), 2)
    0.97

    """
    def __init__(self, embedding_method: BaseEmbedding = GSVD(10), n_neighbors: int = 5,
                 factor_distance: float = 2, leaf_size: int = 16, p: float = 2, tol_nn: float = 0.01, n_jobs: int = 1):
        super(KNN, self).__init__()

        self.embedding_method = embedding_method
        self.n_neighbors = n_neighbors
        self.factor_distance = factor_distance
        self.leaf_size = leaf_size
        self.p = p
        self.tol_nn = tol_nn
        self.n_jobs = n_jobs

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], seeds: Union[np.ndarray, dict]) -> 'KNN':
        """Node classification by k-nearest neighbors in the embedding space.

        Parameters
        ----------
        adjacency :
            Adjacency or biadjacency matrix of the graph.
        seeds :
            Seed nodes. Can be a dict {node: label} or an array where "-1" means no label.

        Returns
        -------
        self: :class:`KNN`
        """
        n = adjacency.shape[0]
        labels = check_seeds(seeds, n).astype(int)
        index_seed = np.argwhere(labels >= 0).ravel()
        index_remain = np.argwhere(labels < 0).ravel()
        labels_seed = labels[index_seed]

        n_neighbors = self.n_neighbors
        if n_neighbors > len(labels_seed):
            warnings.warn(Warning("The number of neighbors cannot exceed the number of seeds. Changed accordingly."))
            n_neighbors = len(labels_seed)

        embedding = self.embedding_method.fit_transform(adjacency)
        embedding_seed = embedding[index_seed]
        embedding_remain = embedding[index_remain]

        tree = cKDTree(embedding_seed, self.leaf_size)
        distances, neighbors = tree.query(embedding_remain, n_neighbors, self.tol_nn, self.p, n_jobs=self.n_jobs)

        if n_neighbors == 1:
            distances = distances[:, np.newaxis]
            neighbors = neighbors[:, np.newaxis]

        labels_neighbor = labels_seed[neighbors]
        index = np.min(distances, axis=1) == 0
        weights_neighbor = np.zeros_like(distances).astype(float)
        # take all seeds at distance zero, if any
        weights_neighbor[index] = distances[index] == 0
        # assign weights with respect to distances for other
        weights_neighbor[~index] = 1 / np.power(distances[~index], self.factor_distance)

        # form the corresponding matrix
        row = list(np.repeat(index_remain, n_neighbors))
        col = list(labels_neighbor.ravel())
        data = list(weights_neighbor.ravel())

        row += list(index_seed)
        col += list(labels_seed)
        data += list(np.ones_like(index_seed))

        membership = normalize(sparse.csr_matrix((data, (row, col)), shape=(n, np.max(labels_seed) + 1)))

        labels = np.zeros(n, dtype=int)
        for i in range(n):
            labels_neighbor = membership[i].indices
            weights_neighbor = membership[i].data
            labels[i] = labels_neighbor[np.argmax(weights_neighbor)]

        self.membership_ = membership
        self.labels_ = labels

        return self


class BiKNN(BaseClassifier):
    """Node classification by K-nearest neighbors in the embedding space.

    Bigraphs.

    Parameters
    ----------
    embedding_method :
        Which algorithm to use to project the nodes in vector space. Default is ``GSVD``.
    n_neighbors :
        Number of nearest neighbors to consider.
    factor_distance :
        Power weighting factor :math:``\\alpha`` applied to the distance to each neighbor.
        Neighbor at distance :math:``d`` has weight :math:``1 / d^\\alpha``. Default is 2.
    leaf_size :
        Leaf size passed to KDTree.
    p :
        Which Minkowski p-norm to use. Default is 2 (Euclidean distance).
    tol_nn :
        Tolerance in nearest neighbors search; the k-th returned value is guaranteed to be no further
        than ``1 + tol_nn`` times the distance to the actual k-th nearest neighbor.
    n_jobs :
        Number of jobs to schedule for parallel processing. If -1 is given all processors are used.

    Attributes
    ----------
    labels_ : np.ndarray
        Label of each row.
    labels_row_ : np.ndarray
        Label of each row (copy of **labels_**).
    labels_col_ : np.ndarray
        Label of each column.
    membership_ : sparse.csr_matrix
        Membership matrix of rows.
    membership_row_ : sparse.csr_matrix
        Membership matrix of rows (copy of **membership_**).
    membership_col_ : sparse.csr_matrix
        Membership matrix of columns.

    Example
    -------
    >>> from sknetwork.data import movie_actor
    >>> biknn = BiKNN(n_neighbors=2)
    >>> graph = movie_actor(metadata=True)
    >>> biadjacency = graph.biadjacency
    >>> seeds_row = {0: 0, 1: 2, 2: 1}
    >>> len(biknn.fit_transform(biadjacency, seeds_row))
    15
    >>> len(biknn.labels_col_)
    16


    """
    def __init__(self, embedding_method: BaseEmbedding = GSVD(10), n_neighbors: int = 5,
                 factor_distance: float = 2, leaf_size: int = 16, p: float = 2, tol_nn: float = 0.01, n_jobs: int = 1):
        super(BiKNN, self).__init__()

        self.embedding_method = embedding_method
        self.n_neighbors = n_neighbors
        self.factor_distance = factor_distance
        self.leaf_size = leaf_size
        self.p = p
        self.tol_nn = tol_nn
        self.n_jobs = n_jobs

        self.labels_row_ = None
        self.labels_col_ = None
        self.membership_row_ = None
        self.membership_col_ = None

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray], seeds_row: Union[np.ndarray, dict],
            seeds_col: Optional[Union[np.ndarray, dict]] = None) -> 'BiKNN':
        """Node classification by k-nearest neighbors in the embedding space.

        Parameters
        ----------
        biadjacency :
            Biadjacency matrix of the graph.
        seeds_row :
            Seed rows. Can be a dict {node: label} or an array where "-1" means no label.
        seeds_col :
            Seed columns (optional). Same format.

        Returns
        -------
        self: :class:`BiKNN`
        """
        n_row, n_col = biadjacency.shape
        labels_row = check_seeds(seeds_row, n_row).astype(int)
        if seeds_col is None:
            labels_col = -np.ones(n_col, dtype=int)
        else:
            labels_col = check_seeds(seeds_col, n_col).astype(int)
        labels = np.hstack((labels_row, labels_col))
        index_seed = np.argwhere(labels >= 0).ravel()
        index_remain = np.argwhere(labels < 0).ravel()
        labels_seed = labels[index_seed]

        n_neighbors = self.n_neighbors
        if n_neighbors > len(labels_seed):
            warnings.warn(Warning("The number of neighbors cannot exceed the number of seeds. Changed accordingly."))
            n_neighbors = len(labels_seed)

        self.embedding_method.fit(biadjacency)
        embedding_row = self.embedding_method.embedding_row_
        embedding_col = self.embedding_method.embedding_col_
        embedding = np.vstack((embedding_row, embedding_col))
        embedding_seed = embedding[index_seed]
        embedding_remain = embedding[index_remain]

        tree = cKDTree(embedding_seed, self.leaf_size)
        distances, neighbors = tree.query(embedding_remain, n_neighbors, self.tol_nn, self.p, n_jobs=self.n_jobs)

        if n_neighbors == 1:
            distances = distances[:, np.newaxis]
            neighbors = neighbors[:, np.newaxis]

        labels_neighbor = labels_seed[neighbors]
        index = np.min(distances, axis=1) == 0
        weights_neighbor = np.zeros_like(distances).astype(float)
        # take all seeds at distance zero, if any
        weights_neighbor[index] = distances[index] == 0
        # assign weights with respect to distances for other
        weights_neighbor[~index] = 1 / np.power(distances[~index], self.factor_distance)

        # form the corresponding matrix
        row = list(np.repeat(index_remain, n_neighbors))
        col = list(labels_neighbor.ravel())
        data = list(weights_neighbor.ravel())

        row += list(index_seed)
        col += list(labels_seed)
        data += list(np.ones_like(index_seed))

        membership = normalize(sparse.csr_matrix((data, (row, col)), shape=(n_row + n_col, np.max(labels_seed) + 1)))

        labels = np.zeros(n_row + n_col, dtype=int)
        for i in range(n_row + n_col):
            labels_neighbor = membership[i].indices
            weights_neighbor = membership[i].data
            labels[i] = labels_neighbor[np.argmax(weights_neighbor)]

        self.labels_row_ = labels[:n_row]
        self.labels_col_ = labels[n_row:]
        self.labels_ = self.labels_row_
        self.membership_row_ = membership[:n_row]
        self.membership_col_ = membership[n_row:]
        self.membership_ = self.membership_row_

        return self
