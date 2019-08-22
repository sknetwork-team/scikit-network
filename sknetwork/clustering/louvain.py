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

from sknetwork import njit
from sknetwork.clustering.post_processing import reindex_clusters
from sknetwork.utils.adjacency_formats import set_adjacency_weights, directed2undirected
from sknetwork.utils.algorithm_base_class import Algorithm
from sknetwork.utils.checks import check_format, check_engine, check_random_state


def membership_matrix(labels: np.ndarray) -> sparse.csr_matrix:
    """
    Builds a n x k matrix of the label assignments, with k the number of labels.

    Parameters
    ----------
    labels :
        Label of each node.

    Returns
    -------
    membership :
        Binary matrix of label assignments.

    """
    n_nodes = len(labels)
    return sparse.csr_matrix((np.ones(n_nodes), (np.arange(n_nodes), labels)))


class AggregateGraph:
    """
    A class of graphs suitable for the Louvain algorithm. Each node represents a cluster.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    out_weights :
        Out-weights.
    in_weights :
        In-weights.

    Attributes
    ----------
    n_nodes : int
        Number of nodes.
    norm_adjacency : sparse.csr_matrix
        Normalized adjacency matrix (sums to 1).
    out_probs : np.ndarray
        Distribution of out-weights (sums to 1).
    in_probs :  np.ndarray
        Distribution of in-weights (sums to 1).
    """

    def __init__(self, adjacency: sparse.csr_matrix, out_weights: np.ndarray, in_weights: np.ndarray):
        self.n_nodes = adjacency.shape[0]
        self.norm_adjacency = adjacency / adjacency.data.sum()
        self.out_probs = out_weights
        self.in_probs = in_weights

    def aggregate(self, row_membership: Union[sparse.csr_matrix, np.ndarray],
                  col_membership: Union[None, sparse.csr_matrix, np.ndarray] = None):
        """
        Aggregates nodes belonging to the same cluster.

        Parameters
        ----------
        row_membership :
            membership matrix (rows).
        col_membership :
            membership matrix (columns).

        Returns
        -------
        The aggregated graph.
        """
        if type(row_membership) == np.ndarray:
            row_membership = membership_matrix(row_membership)

        if col_membership is not None:
            if type(col_membership) == np.ndarray:
                col_membership = membership_matrix(col_membership)

            self.norm_adjacency = row_membership.T.dot(self.norm_adjacency.dot(col_membership)).tocsr()
            self.in_probs = np.array(col_membership.T.dot(self.in_probs).T)

        else:
            self.norm_adjacency = row_membership.T.dot(self.norm_adjacency.dot(row_membership)).tocsr()
            if self.in_probs is not None:
                self.in_probs = np.array(row_membership.T.dot(self.in_probs).T)

        self.out_probs = np.array(row_membership.T.dot(self.out_probs).T)
        self.n_nodes = self.norm_adjacency.shape[0]
        return self


class Optimizer(Algorithm):
    """
    A generic optimization algorithm.

    Attributes
    ----------
    score_ : float
        Total increase of the objective function.
    labels_ : np.ndarray
        Cluster index of each node.
    """

    def __init__(self):
        self.score_ = None
        self.labels_ = None

    def fit(self, graph: AggregateGraph):
        """
        Fit the clusters to the objective function.

         Parameters
         ----------
         graph :
             Graph to cluster.

         Returns
         -------
         self : :class:̀Optimizer`

         """
        return self


@njit
def fit_core(resolution: float, tol: float, n_nodes: int, out_node_probs: np.ndarray,
             in_node_probs: np.ndarray, self_loops: np.ndarray, data: np.ndarray, indices: np.ndarray,
             indptr: np.ndarray) -> (np.ndarray, float):
    """
    Fit the clusters to the objective function.

    Parameters
    ----------
    resolution :
        Resolution parameter (positive).
    tol :
        Minimum increase in modularity to enter a new optimization pass.
    n_nodes :
        Number of nodes.
    out_node_probs :
        Distribution of node weights based on their out-edges (sums to 1).
    in_node_probs :
        Distribution of node weights based on their in-edges (sums to 1).
    self_loops :
        Weights of self loops.
    data :
        CSR format data array of the normalized adjacency matrix.
    indices :
        CSR format index array of the normalized adjacency matrix.
    indptr :
        CSR format index pointer array of the normalized adjacency matrix.

    Returns
    -------
    labels :
        Cluster index of each node.
    total_increase :
        Score of the clustering (total increase in modularity).
    """
    increase: bool = True
    total_increase: float = 0

    labels: np.ndarray = np.arange(n_nodes)
    out_clusters_weights: np.ndarray = out_node_probs.copy()
    in_clusters_weights: np.ndarray = in_node_probs.copy()

    nodes = np.arange(n_nodes)
    while increase:
        increase = False
        pass_increase: float = 0

        for node in nodes:
            node_cluster = labels[node]
            neighbors: np.ndarray = indices[indptr[node]:indptr[node + 1]]
            weights: np.ndarray = data[indptr[node]:indptr[node + 1]]

            neighbor_clusters_weights = np.zeros(n_nodes)
            for i, neighbor in enumerate(neighbors):
                neighbor_clusters_weights[labels[neighbor]] += weights[i]

            unique_clusters = set(labels[neighbors])
            unique_clusters.discard(node_cluster)

            out_ratio = resolution * out_node_probs[node]
            in_ratio = resolution * in_node_probs[node]
            if len(unique_clusters):
                exit_delta: float = 2 * (neighbor_clusters_weights[node_cluster] - self_loops[node])
                exit_delta -= out_ratio * (in_clusters_weights[node_cluster] - in_node_probs[node])
                exit_delta -= in_ratio * (out_clusters_weights[node_cluster] - out_node_probs[node])

                best_delta: float = 0
                best_cluster = node_cluster

                for cluster in unique_clusters:
                    delta: float = 2 * neighbor_clusters_weights[cluster]
                    delta -= out_ratio * in_clusters_weights[cluster]
                    delta -= in_ratio * out_clusters_weights[cluster]

                    local_delta = delta - exit_delta
                    if local_delta > best_delta:
                        best_delta = local_delta
                        best_cluster = cluster

                if best_delta > 0:
                    pass_increase += best_delta
                    out_clusters_weights[node_cluster] -= out_node_probs[node]
                    in_clusters_weights[node_cluster] -= in_node_probs[node]
                    out_clusters_weights[best_cluster] += out_node_probs[node]
                    in_clusters_weights[best_cluster] += in_node_probs[node]
                    labels[node] = best_cluster

        total_increase += pass_increase
        if pass_increase > tol:
            increase = True
    return labels, total_increase


class GreedyModularity(Optimizer):
    """
    A greedy modularity optimizer.

    See :class:`modularity`

    Attributes
    ----------
    resolution : float
        Resolution parameter.
    tol : float
        Minimum modularity increase to enter a new optimization pass.
    engine : str
        ``'default'``, ``'python'`` or ``'numba'``. If ``'default'``, test if numba is available.

    """

    def __init__(self, resolution: float = 1, tol: float = 1e-3, engine: str = 'default'):
        Optimizer.__init__(self)
        self.resolution = resolution
        self.tol = tol
        self.engine = check_engine(engine)

    def fit(self, graph: AggregateGraph):
        """
        Local optimization of modularity.

        Parameters
        ----------
        graph :
            The graph to cluster.

        Returns
        -------
        self : :class:`Optimizer`
        """

        out_node_probs = graph.out_probs
        in_node_probs = graph.in_probs

        adjacency = 0.5 * directed2undirected(graph.norm_adjacency)

        self_loops = adjacency.diagonal()

        indptr: np.ndarray = adjacency.indptr
        indices: np.ndarray = adjacency.indices
        data: np.ndarray = adjacency.data

        if self.engine == 'python':
            increase: bool = True
            total_increase: float = 0.
            labels: np.ndarray = np.arange(graph.n_nodes)

            out_clusters_weights: np.ndarray = out_node_probs.copy()
            in_clusters_weights: np.ndarray = in_node_probs.copy()

            while increase:
                increase = False
                pass_increase: float = 0.

                for node in range(graph.n_nodes):
                    node_cluster: int = labels[node]
                    neighbors: np.ndarray = indices[indptr[node]:indptr[node + 1]]
                    weights: np.ndarray = data[indptr[node]:indptr[node + 1]]

                    neighbors_clusters: np.ndarray = labels[neighbors]
                    unique_clusters: list = list(set(neighbors_clusters.tolist()) - {node_cluster})
                    n_clusters: int = len(unique_clusters)

                    out_ratio = self.resolution * out_node_probs[node]
                    in_ratio = self.resolution * in_node_probs[node]
                    if n_clusters > 0:
                        exit_delta: float = 2 * (weights[labels[neighbors] == node_cluster].sum() - self_loops[node])
                        exit_delta -= out_ratio * (in_clusters_weights[node_cluster] - in_node_probs[node])
                        exit_delta -= in_ratio * (out_clusters_weights[node_cluster] - out_node_probs[node])

                        local_delta: np.ndarray = np.full(n_clusters, -exit_delta)

                        for index_cluster, cluster in enumerate(unique_clusters):
                            delta: float = 2 * weights[labels[neighbors] == cluster].sum()
                            delta -= out_ratio * in_clusters_weights[cluster]
                            delta -= in_ratio * out_clusters_weights[cluster]

                            local_delta[index_cluster] += delta

                        delta_argmax: int = local_delta.argmax()
                        best_delta: float = local_delta[delta_argmax]
                        if best_delta > 0:
                            pass_increase += best_delta
                            best_cluster = unique_clusters[delta_argmax]

                            out_clusters_weights[node_cluster] -= out_node_probs[node]
                            in_clusters_weights[node_cluster] -= in_node_probs[node]
                            out_clusters_weights[best_cluster] += out_node_probs[node]
                            in_clusters_weights[best_cluster] += in_node_probs[node]
                            labels[node] = best_cluster

                total_increase += pass_increase
                if pass_increase > self.tol:
                    increase = True

            self.score_ = total_increase
            _, self.labels_ = np.unique(labels, return_inverse=True)

            return self

        elif self.engine == 'numba':
            labels, total_increase = fit_core(self.resolution, self.tol, graph.n_nodes,
                                              out_node_probs, in_node_probs, self_loops, data, indices, indptr)

            self.score_ = total_increase
            _, self.labels_ = np.unique(labels, return_inverse=True)

            return self

        else:
            raise ValueError('Unknown engine.')


class Louvain(Algorithm):
    """
    Louvain algorithm for graph clustering in Python (default) and Numba.

    Compute the best partition of the nodes with respect to the optimization criterion
    (default: :class:`GreedyModularity`).

    Parameters
    ----------
    engine :
        ``'default'``, ``'python'`` or ``'numba'``. If ``'default'``, tests if numba is available.
    algorithm :
        The optimization algorithm.
        Requires a fit method.
        Requires `score\\_`  and `labels\\_` attributes.

        If ``'default'``, uses greedy modularity optimization algorithm: :class:`GreedyModularity`.
    resolution :
        Resolution parameter.
    weights :
            Weights of nodes.
            ``'degree'`` (default), ``'uniform'``.
    secondary_weights :
        Weights of secondary nodes (for bipartite graphs).
        ``None`` (default), ``'degree'``, ``'uniform'``.
        If ``None``, taken equal to weights.
    tol :
        Minimum increase in the objective function to enter a new optimization pass.
    agg_tol :
        Minimum increase in the objective function to enter a new aggregation pass.
    max_agg_iter :
        Maximum number of aggregations.
        A negative value is interpreted as no limit.
    shuffle_nodes :
        Enables node shuffling before optimization.
    force_undirected : bool (default= ``False``)
            If ``True``, consider the graph as undirected.
    sorted_cluster :
            If ``True``, sort labels in decreasing order of cluster size.
    random_state :
        Random number generator or random seed. If None, numpy.random is used.
    verbose :
        Verbose mode.

    Attributes
    ----------
    labels_ : np.ndarray
        Label of each node.
    secondary_labels_ : np.ndarray
        Label of each secondary node (for bipartite graphs).
    iteration_count_ : int
        Total number of aggregations performed.
    aggregate_graph_ : sparse.csr_matrix
        Adjacency matrix of the aggregate graph at the end of the algorithm.

    Example
    -------
    >>> louvain = Louvain('python')
    >>> adjacency = sparse.identity(3, format='csr')
    >>> louvain.fit(adjacency).labels_
    array([0, 1, 2])

    References
    ----------
    * Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008).
      Fast unfolding of communities in large networks.
      Journal of statistical mechanics: theory and experiment, 2008
      https://arxiv.org/abs/0803.0476

    * Dugué, N., & Perez, A. (2015).
      Directed Louvain: maximizing modularity in directed networks
      (Doctoral dissertation, Université d'Orléans).
      https://hal.archives-ouvertes.fr/hal-01231784/document

    """

    def __init__(self, engine: str = 'default', algorithm: Union[str, Optimizer] = 'default', resolution: float = 1,
                 weights: str = 'degree', secondary_weights: Union[None, str] = None,
                 tol: float = 1e-3, agg_tol: float = 1e-3, max_agg_iter: int = -1, shuffle_nodes: bool = False,
                 force_undirected: bool = False, sorted_cluster: bool = True,
                 random_state: Optional[Union[np.random.RandomState, int]] = None,
                 verbose: bool = False):

        self.random_state = check_random_state(random_state)
        if algorithm == 'default':
            self.algorithm = GreedyModularity(resolution, tol, engine=check_engine(engine))
        elif isinstance(algorithm, Optimizer):
            self.algorithm = algorithm
        else:
            raise TypeError('Algorithm must be \'auto\' or a valid algorithm.')
        self.weights = weights
        self.secondary_weights = secondary_weights
        if type(max_agg_iter) != int:
            raise TypeError('The maximum number of iterations must be an integer.')
        self.agg_tol = agg_tol
        self.max_agg_iter = max_agg_iter
        self.shuffle_nodes = shuffle_nodes
        self.force_undirected = force_undirected
        self.sorted_cluster = sorted_cluster
        self.verbose = verbose
        self.labels_ = None
        self.secondary_labels_ = None
        self.iteration_count_ = None
        self.aggregate_graph_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], custom_weights: Union[None, np.ndarray] = None,
            custom_secondary_weights: Union[None, np.ndarray] = None, force_biadjacency: bool = False) -> 'Louvain':
        """
        Clustering using chosen Optimizer.

        Parameters
        ----------
        adjacency :
            Adjacency or biadjacency matrix of the graph.
        custom_weights :
            Array of input dependent node weights.
        custom_secondary_weights :
            Array of input dependent weights of secondary nodes.
        force_biadjacency :
            If ``True``, force the input matrix to be considered as a biadjacency matrix.

        Returns
        -------
        self: :class:`Louvain`
        """
        adjacency = check_format(adjacency)
        n1, n2 = adjacency.shape
        if custom_weights:
            weights = custom_weights
        else:
            weights = self.weights
        if custom_secondary_weights:
            secondary_weights = custom_secondary_weights
        else:
            secondary_weights = self.secondary_weights
        adjacency, out_weights, in_weights = set_adjacency_weights(adjacency, weights, secondary_weights,
                                                                   self.force_undirected, force_biadjacency)
        n = adjacency.shape[0]
        nodes = np.arange(n)
        if self.shuffle_nodes:
            nodes = self.random_state.permutation(nodes)
            adjacency = adjacency[nodes, :].tocsc()[:, nodes].tocsr()

        graph = AggregateGraph(adjacency, out_weights, in_weights)

        membership = sparse.identity(n, format='csr')
        increase = True
        iteration_count = 0
        if self.verbose:
            print("Starting with", graph.n_nodes, "nodes.")
        while increase:
            iteration_count += 1

            self.algorithm.fit(graph)

            if self.algorithm.score_ <= self.agg_tol:
                increase = False
            else:
                agg_membership = membership_matrix(self.algorithm.labels_)
                membership = membership.dot(agg_membership)
                graph.aggregate(agg_membership)

                if graph.n_nodes == 1:
                    break
            if self.verbose:
                print("Iteration", iteration_count, "completed with", graph.n_nodes, "clusters and ",
                      self.algorithm.score_, "increment.")
            if iteration_count == self.max_agg_iter:
                break

        self.iteration_count_ = iteration_count
        self.labels_ = membership.indices
        if self.shuffle_nodes:
            reverse = np.empty(nodes.size, nodes.dtype)
            reverse[nodes] = np.arange(nodes.size)
            self.labels_ = self.labels_[reverse]
        if self.sorted_cluster:
            self.labels_ = reindex_clusters(self.labels_)
        if n > n1:
            self.secondary_labels_ = self.labels_[n1:]
            self.labels_ = self.labels_[:n1]
        self.aggregate_graph_ = graph.norm_adjacency * adjacency.data.sum()

        return self
