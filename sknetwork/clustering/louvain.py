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

from sknetwork.clustering.base import BaseClustering, BaseBiClustering
from sknetwork.clustering.louvain_core import fit_core
from sknetwork.clustering.postprocess import reindex_labels
from sknetwork.utils.check import check_format, check_random_state, check_probs, check_square
from sknetwork.utils.format import bipartite2directed, directed2undirected, bipartite2undirected
from sknetwork.utils.membership import membership_matrix
from sknetwork.utils.verbose import VerboseMixin


class Louvain(BaseClustering, VerboseMixin):
    """Louvain algorithm for clustering graphs by maximization of modularity.

    * Graphs
    * Digraphs

    Parameters
    ----------
    resolution :
        Resolution parameter.
    modularity : str
        Which objective function to maximize. Can be ``'dugue'``, ``'newman'`` or ``'potts'``.
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
        Label of each node.
    membership_ : sparse.csr_matrix
        Membership matrix.
    adjacency_ : sparse.csr_matrix
        Adjacency matrix between clusters.

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
      Journal of statistical mechanics: theory and experiment, 2008

    * Dugué, N., & Perez, A. (2015).
      `Directed Louvain: maximizing modularity in directed networks
      <https://hal.archives-ouvertes.fr/hal-01231784/document>`_
      (Doctoral dissertation, Université d'Orléans).

    * Traag, V. A., Van Dooren, P., & Nesterov, Y. (2011).
      `Narrow scope for resolution-limit-free community detection.
      <https://arxiv.org/pdf/1104.3083.pdf>`_
      Physical Review E, 84(1), 016114.
    """
    def __init__(self, resolution: float = 1, modularity: str = 'dugue', tol_optimization: float = 1e-3,
                 tol_aggregation: float = 1e-3, n_aggregations: int = -1, shuffle_nodes: bool = False,
                 sort_clusters: bool = True, return_membership: bool = True, return_aggregate: bool = True,
                 random_state: Optional[Union[np.random.RandomState, int]] = None, verbose: bool = False):
        super(Louvain, self).__init__(sort_clusters=sort_clusters, return_membership=return_membership,
                                      return_aggregate=return_aggregate)
        VerboseMixin.__init__(self, verbose)

        self.resolution = np.float32(resolution)
        self.modularity = modularity.lower()
        self.tol = np.float32(tol_optimization)
        self.tol_aggregation = tol_aggregation
        self.n_aggregations = n_aggregations
        self.shuffle_nodes = shuffle_nodes
        self.random_state = check_random_state(random_state)

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
    def _aggregate(adjacency_norm, probs_ou, probs_in, membership: Union[sparse.csr_matrix, np.ndarray]):
        """Aggregate nodes belonging to the same cluster.

        Parameters
        ----------
        adjacency_norm :
            the norm of the adjacency
        probs_ou :
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
        probs_ou = np.array(membership.T.dot(probs_ou).T)
        return adjacency_norm, probs_ou, probs_in

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'Louvain':
        """Fit algorithm to the data.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`Louvain`
        """
        adjacency = check_format(adjacency)
        check_square(adjacency)
        n = adjacency.shape[0]

        if self.modularity == 'potts':
            probs_ou = check_probs('uniform', adjacency)
            probs_in = probs_ou.copy()
        elif self.modularity == 'newman':
            probs_ou = check_probs('degree', adjacency)
            probs_in = probs_ou.copy()
        elif self.modularity == 'dugue':
            probs_ou = check_probs('degree', adjacency)
            probs_in = check_probs('degree', adjacency.T)
        else:
            raise ValueError('Unknown modularity function.')

        nodes = np.arange(n)
        if self.shuffle_nodes:
            nodes = self.random_state.permutation(nodes)
            adjacency = adjacency[nodes, :].tocsc()[:, nodes].tocsr()

        adjacency_clust = adjacency / adjacency.data.sum()

        membership = sparse.identity(n, format='csr')
        increase = True
        count_aggregations = 0
        self.log.print("Starting with", n, "nodes.")
        while increase:
            count_aggregations += 1

            labels_clust, pass_increase = self._optimize(adjacency_clust, probs_ou, probs_in)
            _, labels_clust = np.unique(labels_clust, return_inverse=True)

            if pass_increase <= self.tol_aggregation:
                increase = False
            else:
                membership_clust = membership_matrix(labels_clust)
                membership = membership.dot(membership_clust)
                adjacency_clust, probs_ou, probs_in = self._aggregate(adjacency_clust, probs_ou, probs_in,
                                                                      membership_clust)

                n = adjacency_clust.shape[0]
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
            reverse = np.empty(nodes.size, nodes.dtype)
            reverse[nodes] = np.arange(nodes.size)
            labels = labels[reverse]

        self.labels_ = labels
        self._secondary_outputs(adjacency)

        return self


class BiLouvain(Louvain, BaseBiClustering):
    """BiLouvain algorithm for the clustering of bipartite graphs.

    * Bigraphs

    Parameters
    ----------
    resolution :
        Resolution parameter.
    modularity : str
        Which objective function to maximize. Can be ``'dugue'``, ``'newman'`` or ``'potts'``.
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
        If ``True``, return the biadjacency matrix of the graph between clusters.
    random_state :
        Random number generator or random seed. If None, numpy.random is used.
    verbose :
        Verbose mode.

    Attributes
    ----------
    labels_ : np.ndarray
        Labels of the rows.
    labels_row_ : np.ndarray
        Labels of the rows (copy of **labels_**).
    labels_col_ : np.ndarray
        Labels of the columns.
    membership_row_ : sparse.csr_matrix
        Membership matrix of the rows (copy of **membership_**).
    membership_col_ : sparse.csr_matrix
        Membership matrix of the columns. Only valid if **cluster_both** = `True`.
    biadjacency_ : sparse.csr_matrix
        Biadjacency matrix of the aggregate graph between clusters.

    Example
    -------
    >>> from sknetwork.clustering import BiLouvain
    >>> from sknetwork.data import movie_actor
    >>> bilouvain = BiLouvain()
    >>> biadjacency = movie_actor()
    >>> labels = bilouvain.fit_transform(biadjacency)
    >>> len(labels)
    15
    """
    def __init__(self, resolution: float = 1, modularity: str = 'dugue', tol_optimization: float = 1e-3,
                 tol_aggregation: float = 1e-3, n_aggregations: int = -1, shuffle_nodes: bool = False,
                 sort_clusters: bool = True, return_membership: bool = True, return_aggregate: bool = True,
                 random_state: Optional[Union[np.random.RandomState, int]] = None, verbose: bool = False):
        super(BiLouvain, self).__init__(sort_clusters=sort_clusters, return_membership=return_membership,
                                        return_aggregate=return_aggregate, resolution=resolution, modularity=modularity,
                                        tol_optimization=tol_optimization, verbose=verbose,
                                        tol_aggregation=tol_aggregation, n_aggregations=n_aggregations,
                                        shuffle_nodes=shuffle_nodes, random_state=random_state)

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'BiLouvain':
        """Apply the Louvain algorithm to the corresponding directed graph, with adjacency matrix:

        :math:`A  = \\begin{bmatrix} 0 & B \\\\ 0 & 0 \\end{bmatrix}`

        where :math:`B` is the input (biadjacency matrix).

        Parameters
        ----------
        biadjacency:
            Biadjacency matrix of the graph.

        Returns
        -------
        self: :class:`BiLouvain`
        """
        louvain = Louvain(resolution=self.resolution, modularity=self.modularity, tol_aggregation=self.tol_aggregation,
                          n_aggregations=self.n_aggregations, shuffle_nodes=self.shuffle_nodes,
                          sort_clusters=self.sort_clusters, return_membership=self.return_membership,
                          return_aggregate=False, random_state=self.random_state, verbose=self.log.verbose)
        biadjacency = check_format(biadjacency)
        n_row, _ = biadjacency.shape

        if self.modularity == 'dugue':
            adjacency = bipartite2directed(biadjacency)
        else:
            adjacency = bipartite2undirected(biadjacency)
        louvain.fit(adjacency)

        self.labels_ = louvain.labels_
        self._split_vars(n_row)
        self._secondary_outputs(biadjacency)

        return self
