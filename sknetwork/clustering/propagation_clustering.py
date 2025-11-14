#!/usr/bin/env python3
# coding: utf-8
"""
Created on May, 2020
@author: Thomas Bonald <tbonald@enst.fr>
"""
from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.classification.propagation import Propagation
from sknetwork.clustering.base import BaseClustering
from sknetwork.utils.format import check_format, get_adjacency


class PropagationClustering(BaseClustering, Propagation):
    """Clustering by label propagation.

    Parameters
    ----------
    n_iter : int
        Maximum number of iterations (-1 for infinity).
    node_order : str
        * `'random'`: node labels are updated in random order.
        * `'increasing'`: node labels are updated by increasing order of weight.
        * `'decreasing'`: node labels are updated by decreasing order of weight.
        * Otherwise, node labels are updated by index order.
    weighted : bool
        If ``True``, the vote of each neighbor is proportional to the edge weight.
        Otherwise, all votes have weight 1.
    sort_clusters : bool
        If ``True``, sort labels in decreasing order of cluster size.
    return_probs : bool
        If ``True``, return the probability distribution over clusters (soft clustering).
    return_aggregate : bool
        If ``True``, return the aggregate adjacency matrix or biadjacency matrix between clusters.

    Attributes
    ----------
    labels\_ : np.ndarray, shape (n_nodes,)
        Label of each node.
    probs\_ : sparse.csr_matrix, shape (n_nodes, n_labels)
        Probability distribution over labels.
    aggregate\_ : sparse.csr_matrix
        Aggregate adjacency matrix or biadjacency matrix between clusters.

    Example
    -------
    >>> from sknetwork.clustering import PropagationClustering
    >>> from sknetwork.data import karate_club
    >>> propagation = PropagationClustering()
    >>> graph = karate_club(metadata=True)
    >>> adjacency = graph.adjacency
    >>> labels = propagation.fit_predict(adjacency)
    >>> len(set(labels))
    2

    References
    ----------
    Raghavan, U. N., Albert, R., & Kumara, S. (2007).
    `Near linear time algorithm to detect community structures in large-scale networks.
    <https://arxiv.org/pdf/0709.2938.pdf>`_
    Physical review E, 76(3), 036106.
    """
    def __init__(self, n_iter: int = 5, node_order: str = 'decreasing', weighted: bool = True,
                 sort_clusters: bool = True, return_probs: bool = True, return_aggregate: bool = True):
        Propagation.__init__(self, n_iter, node_order, weighted)
        BaseClustering.__init__(self, sort_clusters, return_probs, return_aggregate)
        self.bipartite = None

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray]) -> 'PropagationClustering':
        """Clustering by label propagation.

        Parameters
        ----------
        input_matrix : sparse.csr_matrix, np.ndarray
            Adjacency matrix or biadjacency matrix of the graph.

        Returns
        -------
        self: :class:`PropagationClustering`
        """
        self._init_vars()

        # input
        input_matrix = check_format(input_matrix)
        adjacency, bipartite = get_adjacency(input_matrix)

        # propagation
        Propagation.fit(self, adjacency)

        # output
        _, self.labels_ = np.unique(self.labels_, return_inverse=True)
        if bipartite:
            self._split_vars(input_matrix.shape)
            self.bipartite = True
        self._secondary_outputs(input_matrix)

        return self
