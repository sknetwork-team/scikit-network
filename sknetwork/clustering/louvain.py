#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in November 2018
@author: Nathan de Lara <nathan.delara@polytechnique.org>
@author: Quentin Lutz <qlutz@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""
from typing import Union, Optional

import numpy as np
from scipy import sparse

from sknetwork.clustering.base import BaseClustering
from sknetwork.clustering.louvain_core import optimize_core
from sknetwork.clustering.postprocess import reindex_labels
from sknetwork.utils.check import check_format, check_random_state, get_probs
from sknetwork.utils.format import get_adjacency, directed2undirected
from sknetwork.utils.membership import get_membership
from sknetwork.log import Log


class Louvain(BaseClustering, Log):
    """Louvain algorithm for clustering graphs by maximization of modularity.

    For bipartite graphs, the algorithm maximizes Barber's modularity by default.

    Parameters
    ----------
    resolution :
        Resolution parameter.
    modularity : str
        Type of modularity to maximize. Can be ``'Dugue'``, ``'Newman'`` or ``'Potts'`` (default = ``'dugue'``).
    tol_optimization :
        Minimum increase in modularity to enter a new optimization pass in the local search.
    tol_aggregation :
        Minimum increase in modularity to enter a new aggregation pass.
    n_aggregations :
        Maximum number of aggregations.
        A negative value is interpreted as no limit.
    shuffle_nodes :
        Enables node shuffling before optimization.
    sort_clusters :
        If ``True``, sort labels in decreasing order of cluster size.
    return_probs :
        If ``True``, return the probability distribution over clusters (soft clustering).
    return_aggregate :
        If ``True``, return the adjacency matrix of the graph between clusters.
    random_state :
        Random number generator or random seed. If None, numpy.random is used.
    verbose :
        Verbose mode.

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
    >>> from sknetwork.clustering import Louvain
    >>> from sknetwork.data import karate_club
    >>> louvain = Louvain()
    >>> adjacency = karate_club()
    >>> labels = louvain.fit_predict(adjacency)
    >>> len(set(labels))
    4

    References
    ----------
    * Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008).
      `Fast unfolding of communities in large networks.
      <https://arxiv.org/abs/0803.0476>`_
      Journal of statistical mechanics: theory and experiment, 2008.

    * Dugué, N., & Perez, A. (2015).
      `Directed Louvain: maximizing modularity in directed networks
      <https://hal.archives-ouvertes.fr/hal-01231784/document>`_
      (Doctoral dissertation, Université d'Orléans).

    * Barber, M. J. (2007).
      `Modularity and community detection in bipartite networks
      <https://arxiv.org/pdf/0707.1616>`_
      Physical Review E, 76(6).
    """

    def __init__(self, resolution: float = 1, modularity: str = 'dugue', tol_optimization: float = 1e-3,
                 tol_aggregation: float = 1e-3, n_aggregations: int = -1, shuffle_nodes: bool = False,
                 sort_clusters: bool = True, return_probs: bool = True, return_aggregate: bool = True,
                 random_state: Optional[Union[np.random.RandomState, int]] = None, verbose: bool = False):
        super(Louvain, self).__init__(sort_clusters=sort_clusters, return_probs=return_probs,
                                      return_aggregate=return_aggregate)
        Log.__init__(self, verbose)

        self.labels_ = None
        self.resolution = resolution
        self.modularity = modularity.lower()
        self.tol_optimization = tol_optimization
        self.tol_aggregation = tol_aggregation
        self.n_aggregations = n_aggregations
        self.shuffle_nodes = shuffle_nodes
        self.random_state = check_random_state(random_state)
        self.bipartite = None

    def _optimize(self, labels, adjacency, out_weights, in_weights):
        """One optimization pass of the Louvain algorithm.

        Parameters
        ----------
        labels :
            Labels of nodes.
        adjacency :
            Adjacency matrix.
        out_weights :
            Out-weights of nodes.
        in_weights :
            In-weights of nodes

        Returns
        -------
        labels :
            Labels of nodes after optimization.
        increase :
            Gain in modularity after optimization.
        """
        labels = labels.astype(np.int64)
        indices = adjacency.indices.astype(np.int64)
        indptr = adjacency.indptr.astype(np.int64)
        data = adjacency.data.astype(np.float32)
        out_weights = out_weights.astype(np.float32)
        in_weights = in_weights.astype(np.float32)
        out_cluster_weights = out_weights.copy()
        in_cluster_weights = in_weights.copy()
        cluster_weights = np.zeros_like(out_cluster_weights).astype(np.float32)
        self_loops = adjacency.diagonal().astype(np.float32)
        return optimize_core(labels, indices, indptr, data, out_weights, in_weights, out_cluster_weights,
                             in_cluster_weights, cluster_weights, self_loops, self.resolution, self.tol_optimization)

    @staticmethod
    def _aggregate(labels, adjacency, out_weights, in_weights):
        """Aggregate nodes belonging to the same cluster.

        Parameters
        ----------
        labels :
            Labels of nodes.
        adjacency :
            Adjacency matrix.
        out_weights :
            Out-weights of nodes.
        in_weights :
            In-weights of nodes.

        Returns
        -------
        Aggregate graph (adjacency matrix, out-weights, in-weights).
        """
        membership = get_membership(labels)
        adjacency_ = membership.T.tocsr().dot(adjacency.dot(membership))
        out_weights_ = membership.T.dot(out_weights)
        in_weights_ = membership.T.dot(in_weights)
        return adjacency_, out_weights_, in_weights_

    def _pre_processing(self, input_matrix, force_bipartite):
        """Pre-processing for Louvain.

         Parameters
        ----------
        input_matrix :
            Adjacency matrix or biadjacency matrix of the graph.
        force_bipartite :
            If ``True``, force the input matrix to be considered as a biadjacency matrix even if square.

        Returns
        -------
        adjacency :
            Adjacency matrix.
        out_weights, in_weights :
            Node weights.
        membership :
            Membership matrix (labels).
        index :
            Index of nodes.
        """
        self._init_vars()

        # adjacency matrix
        force_directed = self.modularity == 'dugue'
        adjacency, self.bipartite = get_adjacency(input_matrix, force_directed=force_directed,
                                                  force_bipartite=force_bipartite)

        # shuffling
        n = adjacency.shape[0]
        index = np.arange(n)
        if self.shuffle_nodes:
            index = self.random_state.permutation(index)
            adjacency = adjacency[index][:, index]

        # node weights
        if self.modularity == 'potts':
            out_weights = get_probs('uniform', adjacency)
            in_weights = out_weights.copy()
        elif self.modularity == 'newman':
            out_weights = get_probs('degree', adjacency)
            in_weights = out_weights.copy()
        elif self.modularity == 'dugue':
            out_weights = get_probs('degree', adjacency)
            in_weights = get_probs('degree', adjacency.T)
        else:
            raise ValueError('Unknown modularity function.')

        # normalized, symmetric adjacency matrix (sums to 1)
        adjacency = directed2undirected(adjacency)
        adjacency = adjacency / adjacency.data.sum()

        # cluster membership
        membership = sparse.identity(n, format='csr')

        return adjacency, out_weights, in_weights, membership, index

    def _post_processing(self, input_matrix, membership, index):
        """Post-processing for Louvain.

         Parameters
        ----------
        input_matrix :
            Adjacency matrix or biadjacency matrix of the graph.
        membership :
            Membership matrix (labels).
        index :
            Index of nodes.
        """
        if self.sort_clusters:
            labels = reindex_labels(membership.indices)
        else:
            labels = membership.indices
        if self.shuffle_nodes:
            reverse = np.empty(index.size, index.dtype)
            reverse[index] = np.arange(index.size)
            labels = labels[reverse]
        self.labels_ = labels
        if self.bipartite:
            self._split_vars(input_matrix.shape)
        self._secondary_outputs(input_matrix)

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray], force_bipartite: bool = False) -> 'Louvain':
        """Fit algorithm to data.

        Parameters
        ----------
        input_matrix :
            Adjacency matrix or biadjacency matrix of the graph.
        force_bipartite :
            If ``True``, force the input matrix to be considered as a biadjacency matrix even if square.

        Returns
        -------
        self : :class:`Louvain`
        """
        input_matrix = check_format(input_matrix)
        adjacency, out_weights, in_weights, membership, index = self._pre_processing(input_matrix, force_bipartite)
        n = adjacency.shape[0]
        count = 0
        stop = False
        while not stop:
            count += 1
            labels = np.arange(n)
            labels, increase = self._optimize(labels, adjacency, out_weights, in_weights)
            _, labels = np.unique(labels, return_inverse=True)
            adjacency, out_weights, in_weights = self._aggregate(labels, adjacency, out_weights, in_weights)
            membership = membership.dot(get_membership(labels))
            n = adjacency.shape[0]
            stop = n == 1
            stop |= increase <= self.tol_aggregation
            stop |= count == self.n_aggregations
            self.print_log("Aggregation:", count, " Clusters:", n, " Increase:", increase)
        self._post_processing(input_matrix, membership, index)
        return self
