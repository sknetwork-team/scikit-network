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
    default = 'numba'
except ImportError:
    def njit(func):
        return func
    default = 'python'

import numpy as np
from scipy import sparse
from typing import Union


class NormalizedGraph:
    """
    A class of graphs suitable for the Louvain algorithm. Each node represents a cluster.

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
        """
        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.
        node_probs :
            Distribution of node weights (sums to 1), used in the second term of modularity.
        """
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


class GreedyModularity(Optimizer):
    """
    A greedy modularity optimizer.
    """

    def __init__(self, resolution: float = 1, tol: float = 1e-3, shuffle_nodes: bool = False):
        """

        Parameters
        ----------
        resolution:
             Resolution parameter (positive)
        tol:
            Minimum increase in modularity to enter a new optimization pass.
        shuffle_nodes:
            If true, shuffle the nodes before starting a new optimization pass.
        """
        Optimizer.__init__(self)
        self.resolution = resolution
        self.tol = tol
        self.shuffle_nodes = shuffle_nodes

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
                        # minus the probability to choose a neighbor in the candidate cluster (with resolution factor)
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

        self.score_ = total_increase
        _, self.labels_ = np.unique(labels, return_inverse=True)

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


class GreedyModularityNumba(Optimizer):
    """A greedy modularity optimizer using Numba for enhanced performance.

    Attributes
    ----------
    labels_:
        Cluster index of each node
    score_:
        Score of the clustering (total increase in modularity)

    Notes
    -----
    Tested with Numba v0.40.1.
    """

    def __init__(self, resolution: float = 1, tol: float = 1e-3, shuffle_nodes: bool = False):
        """

        Parameters
        ----------
        resolution:
            Modularity resolution.
        tol:
            Minimum increase in modularity to enter a new optimization pass.
        shuffle_nodes:
            If True, shuffle the nodes before each optimization pass.
        """
        Optimizer.__init__(self)
        self.resolution = resolution
        self.tol = tol
        self.shuffle_nodes = shuffle_nodes

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
        labels, total_increase = fit_core(self.resolution, self.tol, self.shuffle_nodes, graph.n_nodes,
                                          graph.node_probs, graph.norm_adjacency.diagonal(), graph.norm_adjacency.data,
                                          graph.norm_adjacency.indices, graph.norm_adjacency.indptr)

        self.score_ = total_increase
        _, self.labels_ = np.unique(labels, return_inverse=True)

        return self


class Louvain:
    """Louvain algorithm for graph clustering in Python (default) and Numba.

    Parameters
    ----------
    algorithm:
        The optimization algorithm.
        Requires a fit method.
        Requires score_  and labels_ attributes.
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
    >>> louvain_numba = Louvain(algorithm=GreedyModularityNumba())
    >>> (louvain_numba.fit(graph).labels_ == np.array([0, 1, 2])).all()
    True

    References
    ----------
    Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008).
    Fast unfolding of communities in large networks.
    Journal of statistical mechanics: theory and experiment, 2008
    """

    def __init__(self, algorithm: Union[str, Optimizer] = default, resolution: float = 1, tol: float = 1e-3,
                 shuffle_nodes: bool = False, agg_tol: float = 1e-3, max_agg_iter: int = -1, verbose: bool = False):

        if type(algorithm) == str:
            if algorithm == "numba":
                self.algorithm = GreedyModularityNumba(resolution, tol, shuffle_nodes)
            elif algorithm == "python":
                self.algorithm = GreedyModularity(resolution, tol, shuffle_nodes)
            else:
                raise ValueError('Unknown algorithm name.')
        elif isinstance(algorithm, Optimizer):
            self.algorithm = algorithm
        else:
            raise TypeError('Algorithm must be a string ("numba" or "python") or a valid algorithm.')

        if type(max_agg_iter) != int:
            raise TypeError('The maximum number of iterations must be an integer.')
        self.agg_tol = agg_tol
        self.max_agg_iter = max_agg_iter
        self.verbose = verbose
        self.labels_ = None
        self.iteration_count_ = None
        self.aggregate_graph_ = None

    def fit(self, adjacency: sparse.csr_matrix, node_weights: Union[str, np.ndarray] = 'degree') -> 'Louvain':
        """Clustering using chosen Optimizer.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph to cluster.
        node_weights :
            Node weights used in the second term of modularity.

        Returns
        -------
        self: :class:`Louvain`
        """
        if type(adjacency) != sparse.csr_matrix:
            raise TypeError('The adjacency matrix must be in a scipy compressed sparse row (csr) format.')
        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError('The adjacency matrix must be square.')
        if (adjacency != adjacency.T).nnz != 0:
            raise ValueError('The graph cannot be directed. Please fit a symmetric adjacency matrix.')

        n_nodes = adjacency.shape[0]

        if type(node_weights) == np.ndarray:
            if len(node_weights) != n_nodes:
                raise ValueError('The number of node weights must match the number of nodes.')
            else:
                node_weights_vec = node_weights
        elif type(node_weights) == str:
            if node_weights == 'degree':
                node_weights_vec = adjacency.dot(np.ones(n_nodes))
            elif node_weights == 'uniform':
                node_weights_vec = np.ones(n_nodes)
            else:
                raise ValueError('Unknown distribution of node weights.')
        else:
            raise TypeError(
                'Node weights must be a known distribution ("degree" or "uniform" string) or a custom NumPy array.')

        if np.any(node_weights_vec <= 0):
            raise ValueError('All node weights must be positive.')
        else:
            node_weights_vec = node_weights_vec / np.sum(node_weights_vec)

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
        _, self.labels_ = np.unique(self.labels_, return_inverse=True)
        self.aggregate_graph_ = graph.norm_adjacency * adjacency.data.sum()
        return self
