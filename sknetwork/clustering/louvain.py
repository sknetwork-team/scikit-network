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
from sknetwork.utils.check import check_random_state, get_probs
from sknetwork.utils.format import check_format, get_adjacency, directed2undirected
from sknetwork.utils.membership import get_membership
from sknetwork.log import Log


class Louvain(BaseClustering, Log):
    """Louvain algorithm for clustering graphs by maximization of modularity.

    For bipartite graphs, the algorithm maximizes Barber's modularity by default.

    Parameters
    ----------
    resolution :
        Resolution parameter.
    modularity : stction to maximize. Can be ``'Dugue'``, ``'Newman'`` or ``'Potts'`` (default = ``'dugue'``).
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
    labels_ : np.ndarray, shape (n_labels,)
        Label of each node.
    probs_ : sparse.csr_matrix, shape (n_row, n_labels)
        Probability distribution over labels.
    labels_row_, labels_col_ : np.ndarray
        Labels of rows and columns, for bipartite graphs.
    probs_row_, probs_col_ : sparse.csr_matrix, shape (n_row, n_labels)
        Probability distributions over labels for rows and columns (for bipartite graphs).
    aggregate_ : sparse.csr_matrix
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

    def _optimize(self, adjacency, out_weights, in_weights):
        """One optimization pass of the Louvain algorithm.

        Parameters
        ----------
        adjacency :
            Adjacency matrix.
        out_weights :
            Out-weights of nodes.
        in_weights :
            In-weights of nodes

        Returns
        -------
        labels :
            Labels of the nodes after optimization.
        increase :
            Gain in modularity after optimization.
        """
        out_weights_ = out_weights.astype(np.float32)
        in_weights_ = in_weights.astype(np.float32)

        adjacency_ = 0.5 * directed2undirected(adjacency)
        self_loops = adjacency.diagonal().astype(np.float32)

        indptr: np.ndarray = adjacency_.indptr
        indices: np.ndarray = adjacency_.indices
        data: np.ndarray = adjacency_.data.astype(np.float32)

        return optimize_core(self.resolution, self.tol_optimization, out_weights_, in_weights_, self_loops, data,
                             indices, indptr)

    @staticmethod
    def _aggregate(adjacency, out_weights, in_weights, membership):
        """Aggregate nodes belonging to the same cluster.

        Parameters
        ----------
        adjacency :
            Adjacency matrix.
        out_weights :
            Out-weights of nodes.
        in_weights :
            In-weights of nodes
        membership :
            Membership matrix.

        Returns
        -------
        Aggregate graph (adjacency matrix, out-weights, in-weights).
        """
        adjacency_ = (membership.T.dot(adjacency.dot(membership))).tocsr()
        out_weights_ = np.array(membership.T.dot(out_weights).T)
        in_weights_ = np.array(membership.T.dot(in_weights).T)
        return adjacency_, out_weights_, in_weights_

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
        self._init_vars()
        input_matrix = check_format(input_matrix)
        force_directed = self.modularity == 'dugue'
        adjacency, self.bipartite = get_adjacency(input_matrix, force_directed=force_directed,
                                                  force_bipartite=force_bipartite)

        n = adjacency.shape[0]
        index = np.arange(n)
        if self.shuffle_nodes:
            index = self.random_state.permutation(index)
            adjacency = adjacency[index][:, index]

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

        # aggregate adjacency matrix (sums to 1)
        adjacency_ = adjacency / adjacency.data.sum()
        # cluster membership
        membership = sparse.identity(n, format='csr')
        stop = False
        count_aggregations = 0
        self.print_log("Starting with", n, "nodes.")
        while not stop:
            count_aggregations += 1
            labels, increase = self._optimize(adjacency_, out_weights, in_weights)
            _, labels = np.unique(labels, return_inverse=True)

            if increase <= self.tol_aggregation:
                stop = True
            else:
                membership_ = get_membership(labels)
                membership = membership.dot(membership_)
                adjacency_, out_weights, in_weights = self._aggregate(adjacency_, out_weights, in_weights, membership_)

                if adjacency_.shape[0] == 1:
                    break
            self.print_log("Aggregation:", count_aggregations, " Clusters:", n, " Increase:", increase)
            if count_aggregations == self.n_aggregations:
                break

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

        return self
