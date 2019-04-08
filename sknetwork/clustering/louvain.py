#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 2, 2018
@author: Nathan de Lara <ndelara@enst.fr>
@author: Quentin Lutz <qlutz@enst.fr>
@author: Thomas Bonald <tbonald@enst.fr>
"""

try:
    from numba import njit
except ImportError:
    def njit(func):
        return func

import numpy as np
from scipy import sparse
from .utils import *


class NormalizedGraph:
    """
    A class of graphs suitable for the Louvain algorithm. Each node represents a cluster.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    node_probs :
        Distribution of node weights (sums to 1), used in the second term of modularity.

    Attributes
    ----------
    n_nodes: int
        Number of nodes.
    norm_adjacency: sparse.csr_matrix
        Normalized adjacency matrix (sums to 1).
    node_probs : np.ndarray
        Distribution of node weights (sums to 1).
    """

    def __init__(self, adjacency: sparse.csr_matrix, node_probs: np.ndarray):
        self.n_nodes = adjacency.shape[0]
        self.norm_adjacency = adjacency / adjacency.data.sum()
        self.node_probs = node_probs.copy()

    def aggregate(self, membership: sparse.csr_matrix):
        """Aggregates nodes belonging to the same cluster.

        Parameters
        ----------
        membership:
            Entry i,j equals to 1 if node i belongs to cluster j (0 otherwise).

        Returns
        -------
        The aggregated graph
        """
        self.norm_adjacency = membership.T.dot(self.norm_adjacency.dot(membership)).tocsr()
        self.node_probs = np.array(membership.T.dot(self.node_probs).T)
        self.n_nodes = self.norm_adjacency.shape[0]
        return self


class Optimizer:
    """A generic optimization algorithm.

    Attributes
    ----------
    score_: float
        Total increase of the objective function.
    labels_: np.ndarray
        Cluster index of each node.
    """
    def __init__(self):
        self.score_ = None
        self.labels_ = None

    def fit(self, graph: NormalizedGraph):
        """Fit the clusters to the objective function.

         Parameters
         ----------
         graph:
             Graph to cluster.

         Returns
         -------
         self: :class:̀Optimizer`

         """
        return self


@njit
def fit_core(resolution: float, tol: float, shuffle_nodes: bool, n_nodes: int, node_probs: np.ndarray,
             self_loops: np.ndarray, data: np.ndarray, indices: np.ndarray, indptr: np.ndarray) -> (np.ndarray, float):
    """

    Parameters
    ----------
    resolution:
        Resolution parameter (positive).
    tol:
        Minimum increase in modularity to enter a new optimization pass.
    shuffle_nodes:
        If True, shuffle the nodes before starting a new optimization pass.
    n_nodes:
        Number of nodes.
    node_probs:
        Distribution of node weights (sums to 1).
    self_loops:
        Weights of self loops.
    data:
        CSR format data array of the normalized adjacency matrix.
    indices:
        CSR format index array of the normalized adjacency matrix.
    indptr:
        CSR format index pointer array of the normalized adjacency matrix.

    Returns
    -------
    labels:
        Cluster index of each node
    total_increase:
        Score of the clustering (total increase in modularity)
    """
    increase: bool = True
    total_increase: float = 0

    labels: np.ndarray = np.arange(n_nodes)
    cluster_probs: np.ndarray = node_probs.copy()

    nodes = np.arange(n_nodes)
    while increase:
        increase = False
        pass_increase: float = 0

        if shuffle_nodes:
            nodes = np.random.permutation(np.arange(n_nodes))

        for node in nodes:
            node_cluster = labels[node]

            neighbor_cluster_weights = np.zeros(n_nodes)
            for i in range(indptr[node], indptr[node + 1]):
                neighbor_cluster_weights[labels[indices[i]]] += data[i]

            unique_clusters = set(labels[indices[indptr[node]:indptr[node + 1]]])
            unique_clusters.discard(node_cluster)

            if len(unique_clusters):
                node_prob = node_probs[node]
                node_prob_res = resolution * node_prob

                # total weight of edges to neighbors in the same cluster
                out_delta: float = neighbor_cluster_weights[node_cluster] - self_loops[node]
                # minus the probability to choose a neighbor in the same cluster (with resolution factor)
                out_delta -= node_prob_res * (cluster_probs[node_cluster] - node_prob)

                best_delta: float = 0
                best_cluster = node_cluster

                for cluster in unique_clusters:
                    # total weight of edges to neighbors in the candidate cluster
                    in_delta: float = neighbor_cluster_weights[cluster]
                    # minus the probability to choose a neighbor in the candidate cluster (with resolution factor)
                    in_delta -= node_prob_res * cluster_probs[cluster]

                    local_delta = 2 * (in_delta - out_delta)
                    if local_delta > best_delta:
                        best_delta = local_delta
                        best_cluster = cluster

                if best_delta > 0:
                    pass_increase += best_delta
                    cluster_probs[node_cluster] -= node_prob
                    cluster_probs[best_cluster] += node_prob
                    labels[node] = best_cluster

        total_increase += pass_increase
        if pass_increase > tol:
            increase = True
    return labels, total_increase


class GreedyModularity(Optimizer):
    """
    A greedy modularity optimizer.

    Parameters
    ----------
    resolution:
         Resolution parameter (positive)
    tol:
        Minimum increase in modularity to enter a new optimization pass.
    shuffle_nodes:
        If true, shuffle the nodes before starting a new optimization pass.
    engine: str
        ``'default'``, ``'python'`` or ``'numba'``. If ``'default'``, it will test if numba is available.
    """

    def __init__(self, resolution: float = 1, tol: float = 1e-3, shuffle_nodes: bool = False, engine: str = 'default'):
        Optimizer.__init__(self)
        self.resolution = resolution
        self.tol = tol
        self.shuffle_nodes = shuffle_nodes
        self.engine = check_engine(engine)

    def fit(self, graph: NormalizedGraph):
        """Iterate over the nodes to increase modularity.

         Parameters
         ----------
         graph:
             Graph to cluster.

         Returns
         -------
         self: :class:`Optimizer`
        """
        adjacency: sparse.csr_matrix = graph.norm_adjacency
        sym_error = adjacency - adjacency.T
        if np.any(np.abs(sym_error.data) > 1e-10):
            raise ValueError('The graph cannot be directed. Please fit a symmetric adjacency matrix.')

        if self.engine == 'python':
            increase = True
            total_increase: float = 0

            labels: np.ndarray = np.arange(graph.n_nodes)
            node_probs: np.ndarray = graph.node_probs
            cluster_probs = node_probs.copy()
            self_loops: np.ndarray = graph.norm_adjacency.diagonal()

            while increase:
                increase = False
                pass_increase: float = 0

                if self.shuffle_nodes:
                    nodes = np.random.permutation(np.arange(graph.n_nodes))
                else:
                    nodes = range(graph.n_nodes)

                for node in nodes:
                    node_cluster: int = labels[node]
                    neighbor_weights: np.ndarray = graph.norm_adjacency.data[
                                               graph.norm_adjacency.indptr[node]:graph.norm_adjacency.indptr[node + 1]]
                    neighbors: np.ndarray = graph.norm_adjacency.indices[
                                            graph.norm_adjacency.indptr[node]:graph.norm_adjacency.indptr[node + 1]]
                    neighbor_clusters: np.ndarray = labels[neighbors]
                    unique_clusters: list = list(set(neighbor_clusters.tolist()) - {node_cluster})
                    n_clusters = len(unique_clusters)

                    if n_clusters:
                        node_prob: float = node_probs[node]
                        node_prob_res: float = self.resolution * node_prob

                        # total weight of edges to neighbors in the same cluster
                        out_delta: float = neighbor_weights[neighbor_clusters == node_cluster].sum() - self_loops[node]
                        # minus the probability to choose a neighbor in the same cluster (with resolution factor)
                        out_delta -= node_prob_res * (cluster_probs[node_cluster] - node_prob)

                        local_delta: np.ndarray = np.full(n_clusters, -out_delta)

                        for index_cluster, cluster in enumerate(unique_clusters):
                            # total weight of edges to neighbors in the candidate cluster
                            in_delta: float = neighbor_weights[neighbor_clusters == cluster].sum()
                            # minus the probability to choose a neighbor in the candidate cluster
                            # (with resolution factor)
                            in_delta -= node_prob_res * cluster_probs[cluster]

                            local_delta[index_cluster] += in_delta

                        best_ix: int = local_delta.argmax()
                        best_delta: float = 2 * local_delta[best_ix]
                        if best_delta > 0:
                            pass_increase += best_delta
                            best_cluster = unique_clusters[best_ix]

                            cluster_probs[node_cluster] -= node_prob
                            cluster_probs[best_cluster] += node_prob
                            labels[node] = best_cluster

                total_increase += pass_increase
                if pass_increase > self.tol:
                    increase = True

        elif self.engine == 'numba':
            labels, total_increase = fit_core(self.resolution, self.tol, self.shuffle_nodes, graph.n_nodes,
                                              graph.node_probs, graph.norm_adjacency.diagonal(),
                                              graph.norm_adjacency.data,
                                              graph.norm_adjacency.indices, graph.norm_adjacency.indptr)

        else:
            raise ValueError('Unknown engine.')

        self.score_ = total_increase
        _, self.labels_ = np.unique(labels, return_inverse=True)

        return self


@njit
def fit_directed(resolution: float, tol: float, shuffle_nodes: bool, n_nodes: int, ou_node_probs: np.ndarray,
                 in_node_probs: np.ndarray, self_loops: np.ndarray, ou_data: np.ndarray, ou_indices: np.ndarray,
                 ou_indptr: np.ndarray, in_data: np.ndarray, in_indices: np.ndarray,
                 in_indptr: np.ndarray) -> (np.ndarray, float):
    """

    Parameters
    ----------
    resolution:
        Resolution parameter (positive).
    tol:
        Minimum increase in modularity to enter a new optimization pass.
    shuffle_nodes:
        If True, shuffle the nodes before starting a new optimization pass.
    n_nodes:
        Number of nodes.
    ou_node_probs:
        Distribution of node weights based on their out-edges (sums to 1).
    in_node_probs:
        Distribution of node weights based on their in-edges (sums to 1).
    self_loops:
        Weights of self loops.
    ou_data:
        CSR format data array of the normalized adjacency matrix.
    ou_indices:
        CSR format index array of the normalized adjacency matrix.
    ou_indptr:
        CSR format index pointer array of the normalized adjacency matrix.
    in_data:
        CSR format data array of the normalized transposed adjacency matrix.
    in_indices:
        CSR format index array of the normalized transposed adjacency matrix.
    in_indptr:
        CSR format index pointer array of the normalized transposed adjacency matrix.

    Returns
    -------
    labels:
        Cluster index of each node
    total_increase:
        Score of the clustering (total increase in modularity)
    """
    increase: bool = True
    total_increase: float = 0

    labels: np.ndarray = np.arange(n_nodes)
    ou_clusters_weights: np.ndarray = ou_node_probs.copy()
    in_clusters_weights: np.ndarray = in_node_probs.copy()

    nodes = np.arange(n_nodes)
    while increase:
        increase = False
        pass_increase: float = 0

        if shuffle_nodes:
            nodes = np.random.permutation(np.arange(n_nodes))

        for node in nodes:
            node_cluster = labels[node]
            ou_neighbors: np.ndarray = ou_indices[ou_indptr[node]:ou_indptr[node + 1]]
            ou_weights: np.ndarray = ou_data[ou_indptr[node]:ou_indptr[node + 1]]
            in_neighbors: np.ndarray = in_indices[in_indptr[node]:in_indptr[node + 1]]
            in_weights: np.ndarray = in_data[in_indptr[node]:in_indptr[node + 1]]

            ou_neighbor_cluster_weights = np.zeros(n_nodes)
            for i, neighbor in enumerate(ou_neighbors):
                ou_neighbor_cluster_weights[labels[neighbor]] += ou_weights[i]
            in_neighbor_cluster_weights = np.zeros(n_nodes)
            for i, neighbor in enumerate(in_neighbors):
                in_neighbor_cluster_weights[labels[neighbor]] += in_weights[i]

            neighbors = np.array(list(set(list(ou_neighbors) + list(in_neighbors))))
            unique_clusters = set(labels[neighbors])
            unique_clusters.discard(node_cluster)

            if len(unique_clusters):
                # delta for out edges
                ou_delta: float = ou_neighbor_cluster_weights[node_cluster] - self_loops[node]
                ou_delta -= resolution * ou_node_probs[node] * (in_clusters_weights[node_cluster] - in_node_probs[node])

                # delta for in edges
                in_delta: float = in_neighbor_cluster_weights[node_cluster] - self_loops[node]
                in_delta -= resolution * in_node_probs[node] * (ou_clusters_weights[node_cluster] - ou_node_probs[node])

                exit_delta = ou_delta + in_delta
                best_delta: float = 0
                best_cluster = node_cluster

                for cluster in unique_clusters:
                    ou_delta: float = ou_neighbor_cluster_weights[cluster]
                    ou_delta -= resolution * (ou_node_probs[node] * in_clusters_weights[cluster])

                    in_delta: float = in_neighbor_cluster_weights[cluster]
                    in_delta -= resolution * (in_node_probs[node] * ou_clusters_weights[cluster])

                    local_delta = ou_delta + in_delta - exit_delta
                    if local_delta > best_delta:
                        best_delta = local_delta
                        best_cluster = cluster

                if best_delta > 0:
                    pass_increase += best_delta
                    ou_clusters_weights[node_cluster] -= ou_node_probs[node]
                    in_clusters_weights[node_cluster] -= in_node_probs[node]
                    ou_clusters_weights[best_cluster] += ou_node_probs[node]
                    in_clusters_weights[best_cluster] += in_node_probs[node]
                    labels[node] = best_cluster

        total_increase += pass_increase
        if pass_increase > tol:
            increase = True
    return labels, total_increase


class GreedyDirected(Optimizer):
    """
    A greedy directed modularity optimizer.

    Attributes
    ----------
    resolution:
        modularity resolution
    tol:
        minimum bimodularity increase to enter a new optimization pass
    engine: str
        ``'default'``, ``'python'`` or ``'numba'``. If ``'default'``, it will test if numba is available.

    """

    def __init__(self, resolution: float = 1, tol: float = 1e-3, shuffle_nodes: bool = False, engine: str = 'default'):
        Optimizer.__init__(self)
        self.resolution = resolution
        self.tol = tol
        self.shuffle_nodes = shuffle_nodes
        self.engine = check_engine(engine)

    def fit(self, graph: NormalizedGraph):
        """Iterates over the nodes of the graph and moves them to the cluster of highest increase among their neighbors.

        Parameters
        ----------
        graph:
            the graph to cluster

        Returns
        -------
        self: :class:`BiOptimizer`
        """

        self_loops = graph.norm_adjacency.diagonal()
        ou_node_probs = graph.norm_adjacency.dot(np.ones(graph.n_nodes))
        in_node_probs = graph.norm_adjacency.T.dot(np.ones(graph.n_nodes))

        ou_indptr: np.ndarray = graph.norm_adjacency.indptr
        ou_indices: np.ndarray = graph.norm_adjacency.indices
        ou_data: np.ndarray = graph.norm_adjacency.data

        transposed_adjacency = graph.norm_adjacency.T.tocsr()
        in_indptr: np.ndarray = transposed_adjacency.indptr
        in_indices: np.ndarray = transposed_adjacency.indices
        in_data: np.ndarray = transposed_adjacency.data
        del transposed_adjacency

        if self.engine == 'python':
            increase: bool = True
            total_increase: float = 0.
            labels: np.ndarray = np.arange(graph.n_nodes)

            ou_clusters_weights: np.ndarray = ou_node_probs.copy()
            in_clusters_weights: np.ndarray = in_node_probs.copy()

            while increase:
                increase = False
                pass_increase: float = 0.

                for node in range(graph.n_nodes):
                    node_cluster: int = labels[node]
                    ou_neighbors: np.ndarray = ou_indices[ou_indptr[node]:ou_indptr[node + 1]]
                    ou_weights: np.ndarray = ou_data[ou_indptr[node]:ou_indptr[node + 1]]
                    in_neighbors: np.ndarray = in_indices[in_indptr[node]:in_indptr[node + 1]]
                    in_weights: np.ndarray = in_data[in_indptr[node]:in_indptr[node + 1]]

                    neighbors = np.union1d(ou_neighbors, in_neighbors)
                    neighbors_clusters: np.ndarray = labels[neighbors]
                    unique_clusters: list = list(set(neighbors_clusters.tolist()) - {node_cluster})
                    n_clusters: int = len(unique_clusters)

                    if n_clusters > 0:
                        ou_delta: float = ou_weights[labels[ou_neighbors] == node_cluster].sum() - self_loops[node]
                        ou_delta -= self.resolution * ou_node_probs[node] * (in_clusters_weights[node_cluster] -
                                                                             in_node_probs[node])

                        in_delta: float = in_weights[labels[in_neighbors] == node_cluster].sum() - self_loops[node]
                        in_delta -= self.resolution * in_node_probs[node] * (ou_clusters_weights[node_cluster] -
                                                                             ou_node_probs[node])

                        local_delta: np.ndarray = np.full(n_clusters, -(ou_delta + in_delta))

                        for index_cluster, cluster in enumerate(unique_clusters):
                            ou_delta: float = ou_weights[labels[ou_neighbors] == cluster].sum()
                            ou_delta -= self.resolution * (ou_node_probs[node] * in_clusters_weights[cluster])

                            in_delta: float = in_weights[labels[in_neighbors] == cluster].sum()
                            in_delta -= self.resolution * (in_node_probs[node] * ou_clusters_weights[cluster])

                            local_delta[index_cluster] += ou_delta + in_delta

                        delta_argmax: int = local_delta.argmax()
                        best_delta: float = local_delta[delta_argmax]
                        if best_delta > 0:
                            pass_increase += best_delta
                            best_cluster = unique_clusters[delta_argmax]

                            ou_clusters_weights[node_cluster] -= ou_node_probs[node]
                            in_clusters_weights[node_cluster] -= in_node_probs[node]
                            ou_clusters_weights[best_cluster] += ou_node_probs[node]
                            in_clusters_weights[best_cluster] += in_node_probs[node]
                            labels[node] = best_cluster

                total_increase += pass_increase
                if pass_increase > self.tol:
                    increase = True

            self.score_ = total_increase
            _, self.labels_ = np.unique(labels, return_inverse=True)

            return self

        elif self.engine == 'numba':
            labels, total_increase = fit_directed(self.resolution, self.tol, self.shuffle_nodes, graph.n_nodes,
                                                  ou_node_probs, in_node_probs, self_loops, ou_data, ou_indices,
                                                  ou_indptr, in_data, in_indices, in_indptr)

            self.score_ = total_increase
            _, self.labels_ = np.unique(labels, return_inverse=True)

            return self

        else:
            raise ValueError('Unknown engine.')


class Louvain:
    """Louvain algorithm for graph clustering in Python (default) and Numba.

    Parameters
    ----------
    algorithm:
        The optimization algorithm.
        Requires a fit method.
        Requires `score\\_`  and `labels\\_` attributes.

        If ``'default'``, , it will use a greedy bimodularity optimization algorithm: :class:`GreedyModularity`.
    resolution:
        Resolution parameter.
    tol:
        Minimum increase in the objective function to enter a new optimization pass.
    shuffle_nodes:
        If True, shuffle the nodes before each optimization pass.
    agg_tol:
        Minimum increase in the objective function to enter a new aggregation pass.
    max_agg_iter:
        Maximum number of aggregations.
        A negative value is interpreted as no limit.
    verbose:
        Verbose mode.

    Attributes
    ----------
    labels_: np.ndarray
        Cluster index of each node.
    n_clusters_: int
        The number of clusters in the partition.
    iteration_count_: int
        Total number of aggregations performed.
    aggregate_graph_: sparse.csr_matrix
        Aggregated graph at the end of the algorithm.

    Example
    -------
    >>> louvain = Louvain()
    >>> graph = sparse.identity(3, format='csr')
    >>> (louvain.fit(graph).labels_ == np.array([0, 1, 2])).all()
    True
    >>> louvain_numba = Louvain(algorithm=GreedyModularity(engine='numba'))
    >>> (louvain_numba.fit(graph).labels_ == np.array([0, 1, 2])).all()
    True

    References
    ----------
    Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008).
    Fast unfolding of communities in large networks.
    Journal of statistical mechanics: theory and experiment, 2008
    https://arxiv.org/abs/0803.0476
    """

    def __init__(self, algorithm: Union[str, Optimizer] = 'default', resolution: float = 1, tol: float = 1e-3,
                 shuffle_nodes: bool = False, agg_tol: float = 1e-3, max_agg_iter: int = -1, verbose: bool = False):

        if algorithm == 'default':
                self.algorithm = GreedyModularity(resolution, tol, shuffle_nodes, engine='default')
        elif isinstance(algorithm, Optimizer):
            self.algorithm = algorithm
        else:
            raise TypeError('Algorithm must be \'default\' or a valid algorithm.')

        if type(max_agg_iter) != int:
            raise TypeError('The maximum number of iterations must be an integer.')
        self.agg_tol = agg_tol
        self.max_agg_iter = max_agg_iter
        self.verbose = verbose
        self.labels_ = None
        self.n_clusters_ = None
        self.iteration_count_ = None
        self.aggregate_graph_ = None

    def fit(self, adjacency: sparse.csr_matrix, node_weights: Union[str, np.ndarray] = 'degree') -> 'Louvain':
        """Clustering using chosen Optimizer.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph to cluster.
        node_weights :
            Probabilities for node sampling in the null model. ``'degree'``, ``'uniform'`` or custom weights.

        Returns
        -------
        self: :class:`Louvain`
        """
        if type(adjacency) != sparse.csr_matrix:
            raise TypeError('The adjacency matrix must be in a scipy compressed sparse row (csr) format.')
        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError('The adjacency matrix must be square.')

        node_weights_vec = check_weights(node_weights, adjacency)

        graph = NormalizedGraph(adjacency, node_weights_vec)
        membership = sparse.identity(graph.n_nodes, format='csr')
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
                row = np.arange(graph.n_nodes)
                col = self.algorithm.labels_
                data = np.ones(graph.n_nodes)
                agg_membership = sparse.csr_matrix((data, (row, col)))
                membership = membership.dot(agg_membership)
                graph.aggregate(agg_membership)

                if graph.n_nodes == 1:
                    break
            if self.verbose:
                print("Iteration", iteration_count, "completed with", graph.n_nodes, "clusters")
            if iteration_count == self.max_agg_iter:
                break

        self.iteration_count_ = iteration_count
        self.labels_ = membership.indices
        self.n_clusters_ = len(set(self.labels_))
        _, self.labels_ = np.unique(self.labels_, return_inverse=True)
        self.aggregate_graph_ = graph.norm_adjacency * adjacency.data.sum()
        return self
