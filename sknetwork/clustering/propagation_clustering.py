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
from sknetwork.classification.propagation import Propagation
from sknetwork.utils.check import check_format
from sknetwork.utils.format import bipartite2undirected


class PropagationClustering(BaseClustering, Propagation):
    """Clustering by label propagation.

    * Graphs
    * Digraphs

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
    def __init__(self, n_iter: int = 5, node_order: str = 'decreasing', weighted: bool = True,
                 sort_clusters: bool = True, return_membership: bool = True, return_aggregate: bool = True):
        Propagation.__init__(self, n_iter, node_order, weighted)
        BaseClustering.__init__(self, sort_clusters, return_membership, return_aggregate)

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
        Propagation.fit(self, adjacency)
        _, labels = np.unique(self.labels_, return_inverse=True)

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
    def __init__(self, n_iter: int = 5, node_order: str = 'decreasing', weighted: bool = True,
                 sort_clusters: bool = True, return_membership: bool = True, return_aggregate: bool = True):
        super(BiPropagationClustering, self).__init__(n_iter=n_iter, node_order=node_order, weighted=weighted,
                                                      sort_clusters=sort_clusters, return_membership=return_membership,
                                                      return_aggregate=return_aggregate)

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'BiPropagationClustering':
        """Clustering.

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

        propagation = PropagationClustering(self.n_iter, self.node_order, self.weighted)

        self.labels_ = propagation.fit_transform(adjacency)
        self._split_vars(n_row)
        self._secondary_outputs(biadjacency)

        return self
