#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in March 2024
@author: Thomas Bonald <bonald@enst.fr>
@author: Ahmed Zaiou <ahmed.zaiou@capgemini.com>
"""
from typing import Union, Optional

import numpy as np
from scipy import sparse

from sknetwork.clustering import Louvain
from sknetwork.clustering.louvain_core import optimize_core
from sknetwork.clustering.leiden_core import optimize_refine_core
from sknetwork.utils.membership import get_membership
from sknetwork.utils.check import check_random_state
from sknetwork.log import Log


class Leiden(Louvain):
    """Leiden algorithm for clustering graphs by maximization of modularity.
    Compared to the Louvain algorithm, the partition is refined before each aggregation.

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
    >>> from sknetwork.clustering import Leiden
    >>> from sknetwork.data import karate_club
    >>> leiden = Leiden()
    >>> adjacency = karate_club()
    >>> labels = leiden.fit_predict(adjacency)
    >>> len(set(labels))
    4

    References
    ----------
    * Traag, V. A., Waltman, L., & Van Eck, N. J. (2019).
      `From Louvain to Leiden: guaranteeing well-connected communities`, Scientific reports.

    """

    def __init__(self, resolution: float = 1, modularity: str = 'dugue', tol_optimization: float = 1e-3,
                 tol_aggregation: float = 1e-3, n_aggregations: int = -1, shuffle_nodes: bool = False,
                 sort_clusters: bool = True, return_probs: bool = True, return_aggregate: bool = True,
                 random_state: Optional[Union[np.random.RandomState, int]] = None, verbose: bool = False):
        super(Leiden, self).__init__(sort_clusters=sort_clusters, return_probs=return_probs,
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
        """One optimization pass of the Leiden algorithm.

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
        indices = adjacency.indices
        indptr = adjacency.indptr
        data = adjacency.data.astype(np.float32)
        out_weights = out_weights.astype(np.float32)
        in_weights = in_weights.astype(np.float32)
        membership = get_membership(labels)
        out_cluster_weights = membership.T.dot(out_weights)
        in_cluster_weights = membership.T.dot(in_weights)
        cluster_weights = np.zeros_like(out_cluster_weights).astype(np.float32)
        labels = labels.astype(np.int32)
        self_loops = adjacency.diagonal().astype(np.float32)
        return optimize_core(labels, indices, indptr, data, out_weights, in_weights, out_cluster_weights,
                             in_cluster_weights, cluster_weights, self_loops, self.resolution, self.tol_optimization)

    def _optimize_refine(self, labels, labels_refined, adjacency, out_weights, in_weights):
        """Get the refined partition optimizing modularity.

        Parameters
        ----------
        labels :
            Labels of nodes.
        labels_refined :
            Refined labels of nodes.
        adjacency :
            Adjacency matrix.
        out_weights :
            Out-weights of nodes.
        in_weights :
            In-weights of nodes

        Returns
        -------
        labels_refined :
            Refined labels of nodes.
        """
        indices = adjacency.indices
        indptr = adjacency.indptr
        data = adjacency.data.astype(np.float32)
        out_weights = out_weights.astype(np.float32)
        in_weights = in_weights.astype(np.float32)
        membership = get_membership(labels_refined)
        out_cluster_weights = membership.T.dot(out_weights)
        in_cluster_weights = membership.T.dot(in_weights)
        cluster_weights = np.zeros_like(out_cluster_weights).astype(np.float32)
        self_loops = adjacency.diagonal().astype(np.float32)
        labels = labels.astype(np.int32)
        labels_refined = labels_refined.astype(np.int32)
        return optimize_refine_core(labels, labels_refined, indices, indptr, data, out_weights, in_weights,
                                    out_cluster_weights, in_cluster_weights, cluster_weights, self_loops,
                                    self.resolution)

    @staticmethod
    def _aggregate_refine(labels, labels_refined, adjacency, out_weights, in_weights):
        """Aggregate nodes according to refined labels.

        Parameters
        ----------
        labels :
            Labels of nodes.
        labels_refined :
            Refined labels of nodes.
        adjacency :
            Adjacency matrix.
        out_weights :
            Out-weights of nodes.
        in_weights :
            In-weights of nodes.

        Returns
        -------
        Aggregate graph (labels, adjacency matrix, out-weights, in-weights).
        """
        membership = get_membership(labels)
        membership_refined = get_membership(labels_refined)
        adjacency_ = membership_refined.T.tocsr().dot(adjacency.dot(membership_refined))
        out_weights_ = membership_refined.T.dot(out_weights)
        in_weights_ = membership_refined.T.dot(in_weights)
        labels_ = membership_refined.T.tocsr().dot(membership).indices
        return labels_, adjacency_, out_weights_, in_weights_

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray], force_bipartite: bool = False) -> 'Leiden':
        """Fit algorithm to data.

        Parameters
        ----------
        input_matrix :
            Adjacency matrix or biadjacency matrix of the graph.
        force_bipartite :
            If ``True``, force the input matrix to be considered as a biadjacency matrix even if square.

        Returns
        -------
        self : :class:`Leiden`
        """
        adjacency, out_weights, in_weights, membership, index = self._pre_processing(input_matrix, force_bipartite)
        n = adjacency.shape[0]
        labels = np.arange(n)
        count = 0
        stop = False
        while not stop:
            count += 1
            labels, increase = self._optimize(labels, adjacency, out_weights, in_weights)
            _, labels = np.unique(labels, return_inverse=True)
            labels_original = labels.copy()
            labels_refined = np.arange(len(labels))
            labels_refined = self._optimize_refine(labels, labels_refined, adjacency, out_weights, in_weights)
            _, labels_refined = np.unique(labels_refined, return_inverse=True)
            labels, adjacency, out_weights, in_weights = self._aggregate_refine(labels, labels_refined, adjacency,
                                                                                out_weights, in_weights)
            n = adjacency.shape[0]
            stop = n == 1
            stop |= increase <= self.tol_aggregation
            stop |= count == self.n_aggregations
            if stop:
                membership = membership.dot(get_membership(labels_original))
            else:
                membership = membership.dot(get_membership(labels_refined))
            self.print_log("Aggregation:", count, " Clusters:", n, " Increase:", increase)

        self._post_processing(input_matrix, membership, index)
        return self
