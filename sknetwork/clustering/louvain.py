#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 2, 2018
@author: Nathan de Lara <ndelara@enst.fr>
@author: Quentin Lutz <qlutz@enst.fr>
"""

try:
    from numba import njit
except ImportError:
    def njit(func):
        return func

import numpy as np
from scipy import sparse


class NormalizedGraph:
    """
    A class of graphs suitable for the Louvain algorithm.

    Attributes
    ----------
    n_nodes: int
        the number of nodes in the graph
    norm_adj: normalized adjacency matrix (summing to 1)
    node_weights: vector of node weights
    """

    def __init__(self, adj_matrix, node_weights='degree'):
        """

        Parameters
        ----------
        adj_matrix: adjacency matrix of the graph as SciPy sparse matrix
        node_weights: node weights to be used in the second term of the modularity
        """
        self.n_nodes = adj_matrix.shape[0]
        self.norm_adj = adj_matrix / adj_matrix.data.sum()
        if type(node_weights) == np.ndarray:
            if len(node_weights) != self.n_nodes:
                raise ValueError('The number of node weights must match the number of nodes.')
            if any(node_weights < np.zeros(self.n_nodes)):
                raise ValueError('All node weights must be non-negative.')
            else:
                self.node_weights = node_weights
        elif type(node_weights) == str:
            if node_weights == 'degree':
                self.node_weights = self.norm_adj.dot(np.ones(self.n_nodes))
            elif node_weights == 'uniform':
                self.node_weights = np.ones(self.n_nodes) / self.n_nodes
            else:
                raise ValueError('Unknown distribution type.')
        else:
            raise TypeError(
                'Node weights must be a known distribution ("degree" or "uniform" string) or a custom NumPy array.')

    def aggregate(self, membership):
        """
        Aggregates nodes belonging to the same clusters.
        Parameters
        ----------
        membership: scipy sparse matrix of shape n_nodes x n_clusters

        Returns
        -------
        the aggregated graph
        """
        self.norm_adj = membership.T.dot(self.norm_adj.dot(membership)).tocsr()
        self.node_weights = np.array(membership.T.dot(self.node_weights).T)
        self.n_nodes = self.norm_adj.shape[0]
        return self


class GreedyModularity:
    """
    A greedy modularity optimizer.

    Attributes
    ----------
    score_: total increase of modularity after fitting
    labels_: partition of the nodes. labels[node] = cluster_index
    """

    def __init__(self, resolution=1., tol=0., shuffle_nodes=False):
        """

        Parameters
        ----------
        resolution: modularity resolution
        tol: minimum modularity increase to enter a new optimization pass
        shuffle_nodes: whether to shuffle the nodes before beginning an optimization pass
        """
        self.resolution = resolution
        self.tol = tol
        self.shuffle_nodes = shuffle_nodes
        self.score_ = None
        self.labels_ = None

    def fit(self, graph: NormalizedGraph):
        """
        Iterates over the nodes of the graph and moves them to the cluster of highest increase among their neighbors.
        Parameters
        ----------
        graph: the graph to cluster

        Returns
        -------
        self

        """
        increase = True
        total_increase = 0.

        labels: np.ndarray = np.arange(graph.n_nodes)
        clusters_proba: np.ndarray = graph.node_weights.copy()
        self_loops: np.ndarray = graph.norm_adj.diagonal()

        while increase:
            increase = False
            pass_increase = 0.

            if self.shuffle_nodes:
                nodes = np.random.permutation(np.arange(graph.n_nodes))
            else:
                nodes = range(graph.n_nodes)

            for node in nodes:
                node_cluster: int = labels[node]
                node_weights: np.ndarray = graph.norm_adj.data[
                                           graph.norm_adj.indptr[node]:graph.norm_adj.indptr[node + 1]]
                neighbors: np.ndarray = graph.norm_adj.indices[
                                        graph.norm_adj.indptr[node]:graph.norm_adj.indptr[node + 1]]
                neighbors_clusters: np.ndarray = labels[neighbors]
                unique_clusters: list = list(set(neighbors_clusters.tolist()) - {node_cluster})
                n_clusters: int = len(unique_clusters)

                if n_clusters > 0:
                    node_proba: float = graph.node_weights[node]
                    node_ratio: float = self.resolution * node_proba

                    # node_weights of connections to all other nodes in original cluster
                    out_delta: float = (self_loops[node] - node_weights.dot(neighbors_clusters == node_cluster))
                    # proba to choose (node, other_neighbor) among original cluster
                    out_delta += node_ratio * (clusters_proba[node_cluster] - node_proba)

                    local_delta: np.ndarray = np.full(n_clusters, out_delta)

                    for index_cluster, cluster in enumerate(unique_clusters):
                        # node_weights of connections to all other nodes in candidate cluster
                        in_delta: float = node_weights.dot(neighbors_clusters == cluster)
                        # proba to choose (node, other_neighbor) among new cluster
                        in_delta -= node_ratio * clusters_proba[cluster]

                        local_delta[index_cluster] += in_delta

                    best_delta: float = 2 * max(local_delta)
                    if best_delta > 0:
                        pass_increase += best_delta
                        best_cluster = unique_clusters[local_delta.argmax()]

                        clusters_proba[node_cluster] -= node_proba
                        clusters_proba[best_cluster] += node_proba
                        labels[node] = best_cluster

            total_increase += pass_increase
            if pass_increase > self.tol:
                increase = True

        self.score_ = total_increase
        _, self.labels_ = np.unique(labels, return_inverse=True)

        return self


@njit
def fit_core(shuffle_nodes, n_nodes, node_weights, resolution, self_loops, tol, indptr, indices, weights):
    """

    Parameters
    ----------
    shuffle_nodes: if True, a random permutation of the node is done. The natural order is used otherwise
    n_nodes: number of nodes in the graph
    node_weights: the node weights in the graph
    resolution: the resolution for the Louvain modularity
    self_loops: the weights of the self loops for each node
    tol: the minimum desired increase for each maximization pass
    indptr: the indptr array from the Scipy CSR adjacency matrix
    indices: the indices array from the Scipy CSR adjacency matrix
    weights: the data array from the Scipy CSR adjacency matrix

    Returns
    -------
    a tuple consisting of:
        -the labels found by the algorithm
        -the score of the algorithm (total modularity increase)
    """
    increase = True
    total_increase = 0

    labels: np.ndarray = np.arange(n_nodes)
    clusters_proba: np.ndarray = node_weights.copy()

    local_cluster_weights = np.full(n_nodes, 0.0)
    nodes = np.arange(n_nodes)
    while increase:
        increase = False
        pass_increase = 0.

        if shuffle_nodes:
            nodes = np.random.permutation(np.arange(n_nodes))

        for node in nodes:
            node_cluster = labels[node]

            for k in range(indptr[node], indptr[node + 1]):
                local_cluster_weights[labels[indices[k]]] += weights[k]

            unique_clusters = set(labels[indices[indptr[node]:indptr[node + 1]]])
            unique_clusters.discard(node_cluster)

            if len(unique_clusters):
                node_proba = node_weights[node]
                node_ratio = resolution * node_proba

                # neighbors_weights of connections to all other nodes in original cluster
                out_delta = self_loops[node] - local_cluster_weights[node_cluster]

                # proba to choose (node, other_neighbor) among original cluster
                out_delta += node_ratio * (clusters_proba[node_cluster] - node_proba)

                best_delta = 0.0
                best_cluster = node_cluster

                for cluster in unique_clusters:
                    # neighbors_weights of connections to all other nodes in candidate cluster
                    in_delta = local_cluster_weights[
                        cluster]  # np.sum(neighbors_weights[neighbors_clusters == cluster])
                    local_cluster_weights[cluster] = 0.0
                    # proba to choose (node, other_neighbor) among new cluster
                    in_delta -= node_ratio * clusters_proba[cluster]
                    local_delta = 2 * (out_delta + in_delta)
                    if local_delta > best_delta:
                        best_delta = local_delta
                        best_cluster = cluster

                if best_delta > 0:
                    pass_increase += best_delta
                    clusters_proba[node_cluster] -= node_proba
                    clusters_proba[best_cluster] += node_proba
                    labels[node] = best_cluster
            local_cluster_weights[node_cluster] = 0.0

        total_increase += pass_increase
        if pass_increase > tol:
            increase = True
    return labels, total_increase


class GreedyModularityJiT:
    """
    A greedy modularity optimizer using Numba for enhanced performance.

    Tested with Numba v0.40.1.

    Attributes
    ----------
    score_: total increase of modularity after fitting
    labels_: partition of the nodes. labels[node] = cluster_index
    """

    def __init__(self, resolution=1., tol=0., shuffle_nodes=False):
        """

        Parameters
        ----------
        resolution: modularity resolution
        tol: minimum modularity increase to enter a new optimization pass
        shuffle_nodes: whether to shuffle the nodes before beginning an optimization pass
        """
        self.resolution = resolution
        self.tol = tol
        self.shuffle_nodes = shuffle_nodes
        self.score_ = None
        self.labels_ = None

    def fit(self, graph: NormalizedGraph):
        """
        Iterates over the nodes of the graph and moves them to the cluster of highest increase among their neighbors.
        Parameters
        ----------
        graph: the graph to cluster

        Returns
        -------
        self

        """
        self_loops: np.ndarray = graph.norm_adj.diagonal()

        res_labels, total_increase = fit_core(self.shuffle_nodes,
                                              graph.n_nodes,
                                              graph.node_weights,
                                              self.resolution,
                                              self_loops,
                                              self.tol,
                                              graph.norm_adj.indptr,
                                              graph.norm_adj.indices,
                                              graph.norm_adj.data)

        self.score_ = total_increase
        _, self.labels_ = np.unique(res_labels, return_inverse=True)

        return self


class Louvain:
    """
    Macro algorithm for Louvain clustering.

    Several versions of the Greedy Modularity Maximization are available.
    Those include a pure Python version which is used by default.
    A Numba version named 'GreedyModularityJiT' is also available.

    Attributes
    ----------
    labels_: partition of the nodes. labels[node] = cluster_index
    iteration_count_: number of aggregations performed during the last run of the "fit" method

    Example
    -------
    >>> louvain = Louvain()
    >>> graph = sparse.identity(3, format='csr')
    >>> (louvain.fit(graph).labels_ == np.array([0, 1, 2])).all()
    True
    >>> louvain_jit = Louvain(algorithm=GreedyModularityJiT())
    >>> (louvain_jit.fit(graph).labels_ == np.array([0, 1, 2])).all()
    True

    References
    ----------
    Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008).
    Fast unfolding of communities in large networks.
    Journal of statistical mechanics: theory and experiment, 2008
    """

    def __init__(self, algorithm=GreedyModularity(), tol=0., max_agg_iter: int = -1, verbose=0):
        """

        Parameters
        ----------
        algorithm: the fixed level optimization algorithm, requires a fit method and score_ and labels_ attributes.
        tol: the minimum modularity increase to keep aggregating.
        max_agg_iter: the maximum number of aggregations to perform, a negative value is interpreted as no limit
        verbose: enables verbosity
        """
        self.algorithm = algorithm
        self.tol = tol
        if type(max_agg_iter) != int:
            raise TypeError('The maximum number of iterations must be a integer')
        self.max_agg_iter = max_agg_iter
        self.verbose = verbose
        self.labels_ = None
        self.iteration_count_ = None

    def fit(self, adj_matrix: sparse.csr_matrix, node_weights="degree"):
        """
        Alternates local optimization and aggregation until convergence.
        Parameters
        ----------
        adj_matrix: adjacency matrix of the graph to cluster
        node_weights: node node_weights distribution to be used in the second term of the modularity

        Returns
        -------
        self
        """
        if type(adj_matrix) != sparse.csr_matrix:
            raise TypeError('The adjacency matrix must be in a scipy compressed sparse row (csr) format.')
        # check that the graph is not directed
        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError('The adjacency matrix must be square.')
        if (adj_matrix != adj_matrix.T).nnz != 0:
            raise ValueError('The graph must not be directed. Please fit a symmetric adjacency matrix.')
        graph = NormalizedGraph(adj_matrix, node_weights)
        membership = sparse.identity(graph.n_nodes, format='csr')
        increase = True
        iteration_count = 0
        if self.verbose:
            print("Starting with", graph.n_nodes, "nodes")
        while increase:
            iteration_count += 1
            self.algorithm.fit(graph)
            if self.algorithm.score_ <= self.tol:
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
        return self
