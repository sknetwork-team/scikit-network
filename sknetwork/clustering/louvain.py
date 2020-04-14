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
from sknetwork.utils.format import bipartite2directed, directed2undirected
from sknetwork.utils.check import check_format, check_random_state, check_probs, is_square
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
    return_adjacency :
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
    """
    def __init__(self, resolution: float = 1, tol_optimization: float = 1e-3, tol_aggregation: float = 1e-3,
                 n_aggregations: int = -1, shuffle_nodes: bool = False, sort_clusters: bool = True,
                 return_membership: bool = True, return_adjacency: bool = True,
                 random_state: Optional[Union[np.random.RandomState, int]] = None, verbose: bool = False):
        super(Louvain, self).__init__(sort_clusters=sort_clusters, return_membership=return_membership,
                                      return_adjacency=return_adjacency)
        VerboseMixin.__init__(self, verbose)

        self.random_state = check_random_state(random_state)
        self.tol_aggregation = tol_aggregation
        self.resolution = resolution
        self.tol = tol_optimization
        self.n_aggregations = n_aggregations
        self.shuffle_nodes = shuffle_nodes

    def _optimize(self, n_nodes, adjacency_norm, probs_out, probs_in):
        """One local optimization pass of the Louvain algorithm

        Parameters
        ----------
        n_nodes :
            the number of nodes in the adjacency
        adjacency_norm :
            the norm of the adjacency
        probs_out :
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
        node_probs_in = probs_in
        node_probs_out = probs_out

        adjacency = 0.5 * directed2undirected(adjacency_norm)

        self_loops = adjacency.diagonal()

        indptr: np.ndarray = adjacency.indptr
        indices: np.ndarray = adjacency.indices
        data: np.ndarray = adjacency.data

        return fit_core(self.resolution, self.tol, n_nodes,
                        node_probs_out, node_probs_in, self_loops, data, indices, indptr)

    @staticmethod
    def _aggregate(adjacency_norm, probs_out, probs_in,
                   membership_row: Union[sparse.csr_matrix, np.ndarray],
                   membership_col: Union[None, sparse.csr_matrix, np.ndarray] = None):
        """Aggregate nodes belonging to the same cluster.

        Parameters
        ----------
        adjacency_norm :
            the norm of the adjacency
        probs_out :
            the array of degrees of the adjacency
        probs_in :
            the array of degrees of the transpose of the adjacency
        membership_row :
            membership matrix (rows).
        membership_col :
            membership matrix (columns).

        Returns
        -------
        Aggregate graph.
        """
        if type(membership_row) == np.ndarray:
            membership_row = membership_matrix(membership_row)

        if membership_col is not None:
            if type(membership_col) == np.ndarray:
                membership_col = membership_matrix(membership_col)

            adjacency_norm = membership_row.T.dot(adjacency_norm.dot(membership_col)).tocsr()
            probs_in = np.array(membership_col.T.dot(probs_in).T)

        else:
            adjacency_norm = membership_row.T.dot(adjacency_norm.dot(membership_row)).tocsr()
            if probs_in is not None:
                probs_in = np.array(membership_row.T.dot(probs_in).T)

        probs_out = np.array(membership_row.T.dot(probs_out).T)
        n_nodes = adjacency_norm.shape[0]
        return n_nodes, adjacency_norm, probs_out, probs_in

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
        if not is_square(adjacency):
            raise ValueError('The adjacency matrix is not square. Use BiLouvain() instead.')
        n_nodes = adjacency.shape[0]

        probs_out = check_probs('degree', adjacency)
        probs_in = check_probs('degree', adjacency.T)

        nodes = np.arange(n_nodes)
        if self.shuffle_nodes:
            nodes = self.random_state.permutation(nodes)
            adjacency = adjacency[nodes, :].tocsc()[:, nodes].tocsr()

        adjacency_norm = adjacency / adjacency.data.sum()

        membership = sparse.identity(n_nodes, format='csr')
        increase = True
        count_aggregations = 0
        self.log.print("Starting with", n_nodes, "nodes.")
        while increase:
            count_aggregations += 1

            current_labels, pass_increase = self._optimize(n_nodes, adjacency_norm, probs_out, probs_in)
            _, current_labels = np.unique(current_labels, return_inverse=True)

            if pass_increase <= self.tol_aggregation:
                increase = False
            else:
                membership_agg = membership_matrix(current_labels)
                membership = membership.dot(membership_agg)
                n_nodes, adjacency_norm, probs_out, probs_in = self._aggregate(adjacency_norm, probs_out,
                                                                               probs_in, membership_agg)

                if n_nodes == 1:
                    break
            self.log.print("Aggregation", count_aggregations, "completed with", n_nodes, "clusters and ",
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


class BiLouvain(BaseBiClustering, Louvain):
    """BiLouvain algorithm for the clustering of bipartite graphs.

    * Bigraphs

    Parameters
    ----------
    resolution :
        Resolution parameter.
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
    return_biadjacency :
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

    References
    ----------
    * Dugué, N., & Perez, A. (2015).
      `Directed Louvain: maximizing modularity in directed networks
      <https://hal.archives-ouvertes.fr/hal-01231784/document>`_
      (Doctoral dissertation, Université d'Orléans).
    """

    def __init__(self, resolution: float = 1,
                 tol_optimization: float = 1e-3, tol_aggregation: float = 1e-3, n_aggregations: int = -1,
                 shuffle_nodes: bool = False, sort_clusters: bool = True, return_membership: bool = True,
                 return_biadjacency: bool = True, random_state: Optional[Union[np.random.RandomState, int]] = None,
                 verbose: bool = False):
        BaseBiClustering.__init__(self, sort_clusters=sort_clusters, return_membership=return_membership,
                                  return_biadjacency=return_biadjacency)
        Louvain.__init__(self, resolution=resolution, tol_optimization=tol_optimization, verbose=verbose,
                         tol_aggregation=tol_aggregation, n_aggregations=n_aggregations, shuffle_nodes=shuffle_nodes,
                         sort_clusters=sort_clusters, return_membership=return_membership, random_state=random_state)

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
        louvain = Louvain(resolution=self.resolution, tol_aggregation=self.tol_aggregation,
                          n_aggregations=self.n_aggregations, shuffle_nodes=self.shuffle_nodes,
                          sort_clusters=self.sort_clusters, return_membership=self.return_membership,
                          return_adjacency=False, random_state=self.random_state, verbose=self.log.verbose)
        biadjacency = check_format(biadjacency)
        n_row, _ = biadjacency.shape

        adjacency = bipartite2directed(biadjacency)
        louvain.fit(adjacency)

        self.labels_ = louvain.labels_
        self._split_labels(n_row)
        self._secondary_outputs(biadjacency)

        return self
