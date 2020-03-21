#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
from typing import Union

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from sknetwork.classification import BaseClassifier
from sknetwork.embedding import BaseEmbedding, GSVD
from sknetwork.utils.checks import check_labels, check_seeds


class KNN(BaseClassifier):
    """K-nearest neighbors classifier applied to a graph embedding.

    Parameters
    ----------
    embedding_method:
        Which algorithm to use to project the nodes in vector space. Default is GSVD.
    n_neighbors: int
        Number of neighbors to consider in order to infer label.
    weights: str
        weight function used in prediction. Possible values:
        * ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
        * ‘distance’ : weight points by the inverse of their distance.
    leaf_size: int
        Leaf size passed to KDTree.
        This can affect the speed of the construction and query, as well as the memory required to store the tree.
    p:
        Which Minkowski p-norm to use. 1 is the sum-of-absolute-values “Manhattan” distance,
        2 is the usual Euclidean distance infinity is the maximum-coordinate-difference distance.
        A finite large p may cause a ValueError if overflow can occur.
    eps:
        Return approximate nearest neighbors; the k-th returned value is guaranteed to be no further than (1+eps) times
        the distance to the real k-th nearest neighbor.
    n_jobs:
        Number of jobs to schedule for parallel processing. If -1 is given all processors are used.

    Example
    -------
    >>> from sknetwork.data import karate_club
    >>> clf = KNN(n_neighbors=1)
    >>> adjacency, labels_true = karate_club(return_labels=True)
    >>> seeds = {0: labels_true[0], 33: labels_true[33]}
    >>> labels_pred = clf.fit_transform(adjacency, seeds)
    >>> np.round(np.mean(labels_pred == labels_true), 2)
    0.91

    """
    def __init__(self, embedding_method: BaseEmbedding = GSVD(10, normalize=False), n_neighbors: int = 5,
                 weights: str = 'uniform', leaf_size: int = 16, p=2, eps: float = 0.01, n_jobs=1):
        super(KNN, self).__init__()

        self.embedding_method = embedding_method
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.leaf_size = leaf_size
        self.p = p
        self.eps = eps
        self.n_jobs = n_jobs

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], seeds: Union[np.ndarray, dict]) -> 'KNN':
        """Perform semi-supervised classificationSpectralClustering

        adjacency:
            Adjacency matrix of the graph.
        seeds:
            Labeled seed nodes. Can be a dict {node: label} or an array where "-1" means not labeled.

        Returns
        -------
        self: :class:`KNN`
        """
        seeds_labels: np.ndarray = check_seeds(seeds, adjacency)
        classes, n_classes = check_labels(seeds_labels)
        ix_train = (seeds_labels >= 0)
        ix_test = (seeds_labels < 0)

        x: np.ndarray = self.embedding_method.fit_transform(adjacency)
        x_train = x[ix_train]
        x_test = x[ix_test]

        tree = cKDTree(x_train, self.leaf_size)
        dist, neighbors = tree.query(x_test, self.n_neighbors, self.eps, self.p, n_jobs=self.n_jobs)

        if self.n_neighbors == 1:
            dist = dist[:, np.newaxis]
            neighbors = neighbors[:, np.newaxis]

        neigh_labels: np.ndarray = seeds_labels[neighbors]

        membership: np.ndarray = np.zeros((x_test.shape[0], n_classes))
        if self.weights == 'uniform':
            weights: np.ndarray = np.ones_like(dist)
        elif self.weights == 'distance':
            weights: np.ndarray = 1 / dist
        else:
            raise ValueError('weights must be "uniform" or "distance".')

        for i in range(n_classes):
            tmp = ((neigh_labels == i).astype(float) * weights).sum(axis=1)
            membership[:, i] += tmp

        labels = np.zeros(adjacency.shape[0])
        labels[ix_train] = seeds_labels[ix_train]
        labels[ix_test] = np.argmax(membership, axis=1)
        self.labels_ = np.array([classes[int(val)] for val in labels]).astype(int)

        return self
