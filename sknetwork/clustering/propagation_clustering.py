#!/usr/bin/env python3
# coding: utf-8
"""
Created on May, 2020
@author: Thomas Bonald <tbonald@enst.fr>
"""
from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.clustering import BaseClustering, BaseBiClustering
from sknetwork.classification.vote import vote_update
from sknetwork.utils.check import check_format
from sknetwork.utils.format import bipartite2undirected


class PropagationClustering(BaseClustering):
    """Clustering by label propagation.

    * Graphs
    * Digraphs

    Parameters
    ----------
    n_iter : int
        Maximum number of iterations (-1 for infinity).
    sort_clusters :
            If ``True``, sort labels in decreasing order of cluster size.
    return_membership :
            If ``True``, return the membership matrix of nodes to each cluster (soft clustering).
    return_aggregate :
            If ``True``, return the adjacency matrix of the graph between clusters.

    Attributes
    ----------
    labels_ : np.ndarray
        Label of each node.
    membership_ : sparse.csr_matrix
        Membership matrix (columns = labels).

    Example
    -------
    >>> from sknetwork.clustering import PropagationClustering
    >>> from sknetwork.data import karate_club
    >>> propagation = PropagationClustering()
    >>> graph = karate_club(metadata=True)
    >>> adjacency = graph.adjacency
    >>> labels = propagation.fit_transform(adjacency)
    >>> len(set(labels))
    2

    References
    ----------
    Raghavan, U. N., Albert, R., & Kumara, S. (2007).
    `Near linear time algorithm to detect community structures in large-scale networks.
    <https://arxiv.org/pdf/0709.2938.pdf>`_
    Physical review E, 76(3), 036106.
    """
    def __init__(self, n_iter: int = 5, sort_clusters: bool = True,
                 return_membership: bool = True, return_aggregate: bool = True):
        super(PropagationClustering, self).__init__(sort_clusters=sort_clusters,
                                                    return_membership=return_membership,
                                                    return_aggregate=return_aggregate)

        if n_iter < 0:
            self.n_iter = np.inf
        else:
            self.n_iter = n_iter

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'PropagationClustering':
        """Clustering by label propagation.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`PropagationClustering`
        """
        adjacency = check_format(adjacency)
        n = adjacency.shape[0]

        # nodes labelled in decreasing order of weights
        weights = adjacency.T.dot(np.ones(n))
        index = np.argsort(-weights).astype(np.int32)

        labels = np.arange(n, dtype=np.int32)
        labels_prev = np.zeros(n, dtype=np.int32)

        indptr = adjacency.indptr.astype(np.int32)
        indices = adjacency.indices.astype(np.int32)
        data = adjacency.data.astype(np.float32)

        t = 0
        while t < self.n_iter and not np.array_equal(labels_prev, labels):
            t += 1
            labels_prev = labels.copy()
            labels = vote_update(indptr, indices, data, labels, index)

        _, labels = np.unique(labels, return_inverse=True)

        self.labels_ = labels
        self._secondary_outputs(adjacency)

        return self


class BiPropagationClustering(PropagationClustering, BaseBiClustering):
    """Clustering by label propagation in bipartite graphs.

    * Bigraphs

    Parameters
    ----------
    n_iter :
        Maximum number of iteration (-1 for infinity).
    sort_clusters :
            If ``True``, sort labels in decreasing order of cluster size.
    return_membership :
            If ``True``, return the membership matrix of nodes to each cluster (soft clustering).
    return_aggregate :
        If ``True``, return the biadjacency matrix of the graph between clusters.

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
    >>> from sknetwork.clustering import BiPropagationClustering
    >>> from sknetwork.data import movie_actor
    >>> bipropagation = BiPropagationClustering()
    >>> graph = movie_actor(metadata=True)
    >>> biadjacency = graph.biadjacency
    >>> len(bipropagation.fit_transform(biadjacency))
    15
    >>> len(bipropagation.labels_col_)
    16
    """
    def __init__(self, n_iter: int = 5, sort_clusters: bool = True,
                 return_membership: bool = True, return_aggregate: bool = True):
        super(BiPropagationClustering, self).__init__(n_iter, sort_clusters=sort_clusters,
                                                      return_membership=return_membership,
                                                      return_aggregate=return_aggregate)

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'BiPropagationClustering':
        """Clustering by k-nearest neighbors in the embedding space.

        Parameters
        ----------
        biadjacency :
            Biadjacency matrix of the graph.

        Returns
        -------
        self: :class:`BiPropagationClustering`
        """
        n_row, n_col = biadjacency.shape
        biadjacency = check_format(biadjacency)
        adjacency = bipartite2undirected(biadjacency)

        propagation = PropagationClustering(self.n_iter, self.sort_clusters, return_membership=False,
                                            return_aggregate=False)

        self.labels_ = propagation.fit_transform(adjacency)
        self._split_vars(n_row)
        self._secondary_outputs(biadjacency)

        return self
