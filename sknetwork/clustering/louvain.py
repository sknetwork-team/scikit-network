#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 2, 2018
@author: Nathan de Lara <ndelara@enst.fr>
@author: Quentin Lutz <qlutz@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""
from typing import Union, Optional

import numpy as np
from scipy import sparse

from sknetwork.clustering.base import BaseClustering
from sknetwork.clustering.louvain_core import fit_core
from sknetwork.clustering.postprocess import reindex_labels
from sknetwork.utils.check import check_random_state, get_probs
from sknetwork.utils.format import check_format, get_adjacency, directed2undirected
from sknetwork.utils.membership import membership_matrix
from sknetwork.utils.verbose import VerboseMixin


class Louvain(BaseClustering, VerboseMixin):
    """Louvain algorithm for clustering graphs by maximization of modularity.

    For bipartite graphs, the algorithm maximizes Barber's modularity by default.

    Parameters
    ----------
    resolution :
        Resolution parameter.
    modularity : str
        Which objective function to maximize. Can be ``'dugue'``, ``'newman'`` or ``'potts'`` (default = ``'dugue'``).
    tol_optimization :
        Minimum increase in the objective function to enter a new optimization pass.
    tol_aggregation :
        Minimum increase in the objective function to enter a new aggregation pass.
    n_aggregations :
        Maximum number of aggregations.
        A negative value is interpreted as no limit.
    shuffle_nodes :
        Enables node shuffling before optimization.
    sort_clusters :
        If ``True``, sort labels in decreasing order of cluster size.
    return_membership :
            If ``True``, return the membership matrix of nodes to each cluster (soft clustering).
    return_aggregate :
            If ``True``, return the adjacency matrix of the graph between clusters.
    random_state :
        Random number generator or random seed. If None, numpy.random is used.
    verbose :
        Verbose mode.

    Attributes
    ----------
    labels_ : np.ndarray
        Labels of the nodes.
    labels_row_ : np.ndarray
        Labels of the rows (for bipartite graphs).
    labels_col_ : np.ndarray
        Labels of the columns (for bipartite graphs).
    membership_ : sparse.csr_matrix
        Membership matrix of the nodes, shape (n_nodes, n_clusters).
    membership_row_ : sparse.csr_matrix
        Membership matrix of the rows (for bipartite graphs).
    membership_col_ : sparse.csr_matrix
        Membership matrix of the columns (for bipartite graphs).
    aggregate_ : sparse.csr_matrix
        Aggregate adjacency matrix or biadjacency matrix between clusters.

    Example
    -------
    >>> from sknetwork.clustering import Louvain
    >>> from sknetwork.data import karate_club
    >>> louvain = Louvain()
    >>> adjacency = karate_club()
    >>> labels = louvain.fit_transform(adjacency)
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
                 sort_clusters: bool = True, return_membership: bool = True, return_aggregate: bool = True,
                 random_state: Optional[Union[np.random.RandomState, int]] = None, verbose: bool = False):
        super(Louvain, self).__init__(sort_clusters=sort_clusters, return_membership=return_membership,
                                      return_aggregate=return_aggregate)
        VerboseMixin.__init__(self, verbose)

        self.resolution = resolution
        self.modularity = modularity.lower()
        self.tol = tol_optimization
        self.tol_aggregation = tol_aggregation
        self.n_aggregations = n_aggregations
        self.shuffle_nodes = shuffle_nodes
        self.random_state = check_random_state(random_state)
        self.bipartite = None

    def _optimize(self, adjacency_norm, probs_ou, probs_in):
        """One local optimization pass of the Louvain algorithm

        Parameters
        ----------
        adjacency_norm :
            the norm of the adjacency
        probs_ou :
            the array of degrees of the adjacency
        probs_in :
            the array of degrees of the transpose of the adjacency

        Returns
        -------
        labels :
            the communities of each node after optimization
        pass_increase :
            the increase in modularity gained after optimization
        """
        node_probs_in = probs_in.astype(np.float32)
        node_probs_ou = probs_ou.astype(np.float32)

        adjacency = 0.5 * directed2undirected(adjacency_norm)

        self_loops = adjacency.diagonal().astype(np.float32)

        indptr: np.ndarray = adjacency.indptr
        indices: np.ndarray = adjacency.indices
        data: np.ndarray = adjacency.data.astype(np.float32)

        return fit_core(self.resolution, self.tol, node_probs_ou, node_probs_in, self_loops, data, indices, indptr)

    @staticmethod
    def _aggregate(adjacency_norm, probs_out, probs_in, membership: Union[sparse.csr_matrix, np.ndarray]):
        """Aggregate nodes belonging to the same cluster.

        Parameters
        ----------
        adjacency_norm :
            the norm of the adjacency
        probs_out :
            the array of degrees of the adjacency
        probs_in :
            the array of degrees of the transpose of the adjacency
        membership :
            membership matrix (rows).

        Returns
        -------
        Aggregate graph.
        """
        adjacency_norm = (membership.T.dot(adjacency_norm.dot(membership))).tocsr()
        probs_in = np.array(membership.T.dot(probs_in).T)
        probs_out = np.array(membership.T.dot(probs_out).T)
        return adjacency_norm, probs_out, probs_in

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
        self: :class:`Louvain`
        """
        self._init_vars()
        input_matrix = check_format(input_matrix)
        if self.modularity == 'dugue':
            adjacency, self.bipartite = get_adjacency(input_matrix, force_directed=True,
                                                      force_bipartite=force_bipartite)
        else:
            adjacency, self.bipartite = get_adjacency(input_matrix, force_bipartite=force_bipartite)

        n = adjacency.shape[0]

        index = np.arange(n)
        if self.shuffle_nodes:
            self.random_state.permutation(index)
            adjacency = adjacency[index][:, index]

        if self.modularity == 'potts':
            probs_out = get_probs('uniform', adjacency)
            probs_in = probs_out.copy()
        elif self.modularity == 'newman':
            probs_out = get_probs('degree', adjacency)
            probs_in = probs_out.copy()
        elif self.modularity == 'dugue':
            probs_out = get_probs('degree', adjacency)
            probs_in = get_probs('degree', adjacency.T)
        else:
            raise ValueError('Unknown modularity function.')

        adjacency_cluster = adjacency / adjacency.data.sum()

        membership = sparse.identity(n, format='csr')
        increase = True
        count_aggregations = 0
        self.log.print("Starting with", n, "nodes.")
        while increase:
            count_aggregations += 1

            labels_cluster, pass_increase = self._optimize(adjacency_cluster, probs_out, probs_in)
            _, labels_cluster = np.unique(labels_cluster, return_inverse=True)

            if pass_increase <= self.tol_aggregation:
                increase = False
            else:
                membership_cluster = membership_matrix(labels_cluster)
                membership = membership.dot(membership_cluster)
                adjacency_cluster, probs_out, probs_in = self._aggregate(adjacency_cluster, probs_out, probs_in,
                                                                        membership_cluster)

                n = adjacency_cluster.shape[0]
                if n == 1:
                    break
            self.log.print("Aggregation", count_aggregations, "completed with", n, "clusters and ",
                           pass_increase, "increment.")
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
