#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov, 2019
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <tbonald@enst.fr>
"""

import warnings

from typing import Union

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from sknetwork.classification import BaseClassifier
from sknetwork.embedding import BaseEmbedding, GSVD
from sknetwork.linalg.normalize import normalize
from sknetwork.utils.check import check_seeds


class KNN(BaseClassifier):
    """K-nearest neighbors classifier applied to a graph embedding.

    Parameters
    ----------
    embedding_method :
        Which algorithm to use to project the nodes in vector space. Default is GSVD.
    n_neighbors :
        Number of neighbors to consider in order to infer label.
    factor_distance :
        Power weighting factor :math:``\\alpha`` applied to the distance to each neighbor.
        Neighbor at distance :math:``d`` has weight :math:``1 / d^\\alpha``. Default is 0 (no weighting).
    leaf_size :
        Leaf size passed to KDTree.
        This can affect the speed of the construction and query, as well as the memory required to store the tree.
    p :
        Which Minkowski p-norm to use. 1 is the sum-of-absolute-values “Manhattan” distance,
        2 is the usual Euclidean distance infinity is the maximum-coordinate-difference distance.
        A finite large p may cause a ValueError if overflow can occur.
    eps :
        Return approximate nearest neighbors; the k-th returned value is guaranteed to be no further than (1+eps) times
        the distance to the real k-th nearest neighbor.
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
    >>> from sknetwork.data import KarateClub
    >>> knn = KNN(n_neighbors=1)
    >>> graph = KarateClub()
    >>> adjacency = graph.adjacency
    >>> labels_true = graph.labels
    >>> seeds = {0: labels_true[0], 33: labels_true[33]}
    >>> labels_pred = knn.fit_transform(adjacency, seeds)
    >>> np.round(np.mean(labels_pred == labels_true), 2)
    0.91

    """
    def __init__(self, embedding_method: BaseEmbedding = GSVD(10, normalize=False), n_neighbors: int = 5,
                 factor_distance: float = 2, leaf_size: int = 16, p: float = 2, eps: float = 0.01, n_jobs: int = 1):
        super(KNN, self).__init__()

        self.embedding_method = embedding_method
        self.n_neighbors = n_neighbors
        self.factor_distance = factor_distance
        self.leaf_size = leaf_size
        self.p = p
        self.eps = eps
        self.n_jobs = n_jobs

        self.membership_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], seeds: Union[np.ndarray, dict]) -> 'KNN':
        """Perform semi-supervised classification by k-nearest neighbors.

        adjacency :
            Adjacency or biadjacency matrix of the graph.
        seeds :
            Seed nodes. Can be a dict {node: label} or an array where "-1" means no label.

        Returns
        -------
        self: :class:`KNN`
        """
        n = adjacency.shape[0]
        labels = check_seeds(seeds, adjacency).astype(int)
        index_seed = np.argwhere(labels >= 0).ravel()
        index_test = np.argwhere(labels < 0).ravel()
        labels_seed = labels[index_seed]

        n_neighbors = self.n_neighbors
        if n_neighbors > len(labels_seed):
            warnings.warn(Warning("The number of neighbors cannot exceed the number of seeds. Changed accordingly."))
            n_neighbors = len(labels_seed)

        embedding = self.embedding_method.fit_transform(adjacency)
        embedding_seed = embedding[index_seed]
        embedding_test = embedding[index_test]

        tree = cKDTree(embedding_seed, self.leaf_size)
        distances, neighbors = tree.query(embedding_test, n_neighbors, self.eps, self.p, n_jobs=self.n_jobs)

        if n_neighbors == 1:
            distances = distances[:, np.newaxis]
            neighbors = neighbors[:, np.newaxis]

        labels_neighbor = labels_seed[neighbors]
        index = (np.min(distances, axis=1) == 0)
        weights_neighbor = np.zeros_like(distances).astype(float)
        weights_neighbor[index] = (distances[index] == 0)
        weights_neighbor[~index] = 1 / np.power(distances[~index], self.factor_distance)

        row = list(np.repeat(index_test, n_neighbors))
        col = list(labels_neighbor.ravel())
        data = list(weights_neighbor.ravel())

        row += list(index_seed)
        col += list(labels_seed)
        data += list(np.ones_like(index_seed))

        membership = normalize(sparse.csr_matrix((data, (row, col)), shape=(n, np.max(labels_seed) + 1)))

        labels = np.zeros(n, dtype=int)
        for i in range(n):
            labels_neighbor = membership.indices[membership.indptr[i]: membership.indptr[i + 1]]
            weights_neighbor = membership.data[membership.indptr[i]: membership.indptr[i + 1]]
            labels[i] = labels_neighbor[np.argmax(weights_neighbor)]

        self.membership_ = membership
        self.labels_ = labels

        return self
