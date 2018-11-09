#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 2, 2018
@author: Nathan de Lara <ndelara@enst.fr>
@author: Quentin Lutz <qlutz@enst.fr>
"""

try:
    from numba import jit, njit, prange
except ImportError:
    def njit(func):
        return func

import numpy as np
from scipy import sparse


class NormalizedGraph:
    """
    A class of graph specialized for Louvain algorithm.

    Attributes
    ----------
    n_nodes: the number of nodes in the graph
    norm_adj: normalized adjacency matrix such that the coefficients sum to 1
    node_weights: vector of normalized node degrees
    """

    def __init__(self, adj_matrix, node_weights='degree'):
        """

        Parameters
        ----------
        adj_matrix: adjacency matrix of the graph in a SciPy sparse matrix
        node_weights: node node_weights distribution to be used in the second term of the modularity
        """
        self.n_nodes = adj_matrix.shape[0]
        self.norm_adj = adj_matrix / adj_matrix.sum()
        if type(node_weights) == np.ndarray:
            if len(node_weights) != self.n_nodes:
                raise ValueError('The number of node weights should match the number of nodes.')
            if any(node_weights < np.zeros(self.n_nodes)):
                raise ValueError('All node weights should be non-negative.')
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


@njit
def fit_core(shuffle_nodes, n_nodes, labels, edge_weights,
             adjacency, node_weights, resolution, self_loops, clusters_proba, tol):
    """

    Parameters
    ----------
    shuffle_nodes: if True, a random permutation of the node is done. The natural order is used otherwise
    n_nodes: number of nodes in the graph
    labels: the initial labels
    edge_weights: the edge weights in the graph
    adjacency: the adjacency matrix without weights
    node_weights: the node weights in the graph
    resolution: the resolution for the Louvain modularity
    self_loops: the weights of the self loops for each node
    clusters_proba: the neighbors_weights of each cluster
    tol: the minimum desired increase for each maximization pass

    Returns
    -------
    a tuple consisting of:
        -the labels found by the algorithm
        -the score of the algorithm (total modularity increase)
    """
    increase = True
    total_increase = 0
    while increase:
        increase = False
        pass_increase = 0.

        if shuffle_nodes:
            nodes = np.random.permutation(np.arange(n_nodes))
        else:
            nodes = np.arange(n_nodes)

        for node in nodes:
            node_cluster: int = labels[node]
            neighbors_weights: np.ndarray = edge_weights[node]
            neighbors: np.ndarray = adjacency[node]
            neighbors_clusters: np.ndarray = labels[neighbors]
            unique_clusters: np.ndarray = np.unique(neighbors_clusters)
            unique_clusters = unique_clusters[unique_clusters != node_cluster]
            n_clusters: int = len(unique_clusters)

            if n_clusters > 0:
                node_proba: float = node_weights[node]
                node_ratio: float = resolution * node_proba

                # neighbors_weights of connections to all other nodes in original cluster
                out_delta: float = (self_loops[node]
                                    - neighbors_weights.dot((neighbors_clusters == node_cluster).astype(np.float64)))
                # proba to choose (node, other_neighbor) among original cluster
                out_delta += node_ratio * (clusters_proba[node_cluster] - node_proba)

                local_delta: np.ndarray = np.full(n_clusters, out_delta)

                for cluster_index in range(n_clusters):
                    cluster = unique_clusters[cluster_index]
                    # neighbors_weights of connections to all other nodes in candidate cluster
                    in_delta: float = neighbors_weights.dot((neighbors_clusters == cluster).astype(np.float64))
                    # proba to choose (node, other_neighbor) among new cluster
                    in_delta -= node_ratio * clusters_proba[cluster]

                    local_delta[cluster_index] += in_delta

                best_delta: float = 2 * local_delta.max()
                if best_delta > 0:
                    pass_increase += best_delta
                    best_cluster = unique_clusters[local_delta.argmax()]

                    clusters_proba[node_cluster] -= node_proba
                    clusters_proba[best_cluster] += node_proba
                    labels[node] = best_cluster

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
        labels: np.ndarray = np.arange(graph.n_nodes)
        clusters_proba: np.ndarray = graph.node_weights.copy()
        self_loops: np.ndarray = graph.norm_adj.diagonal()

        adjacency, neighbors_weights = graph.n_nodes * [None], graph.n_nodes * [None]
        for node in range(graph.n_nodes):
            node_row = graph.norm_adj[node]
            adjacency[node]: np.ndarray = node_row.indices
            neighbors_weights[node]: np.ndarray = node_row.data

        res_labels, total_increase = fit_core(self.shuffle_nodes,
                                              graph.n_nodes,
                                              labels,
                                              neighbors_weights,
                                              adjacency,
                                              graph.node_weights,
                                              self.resolution,
                                              self_loops,
                                              clusters_proba,
                                              self.tol)

        self.score_ = total_increase
        self.labels_ = res_labels

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

        adjacency, neighbors_weights = graph.n_nodes * [None], graph.n_nodes * [None]
        for node in range(graph.n_nodes):
            node_row = graph.norm_adj[node]
            adjacency[node]: np.ndarray = node_row.indices
            neighbors_weights[node]: np.ndarray = node_row.data

        while increase:
            increase = False
            pass_increase = 0.

            if self.shuffle_nodes:
                nodes = np.random.permutation(np.arange(graph.n_nodes))
            else:
                nodes = range(graph.n_nodes)

            for node in nodes:
                node_cluster: int = labels[node]
                node_weights: np.ndarray = neighbors_weights[node]
                neighbors: np.ndarray = adjacency[node]
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
        self.labels_ = labels

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
    >>>louvain = Louvain()
    >>>graph = sparse.identity(3, format='csr')
    >>>louvain.fit(graph).labels_
        array([0, 1, 2])
    >>>louvain_jit = Louvain(algorithm=GreedyModularityJiT())
    >>>louvain_jit.fit(graph).labels_
        array([0, 1, 2])
    """

    def __init__(self, algorithm=GreedyModularity(), tol=0., max_agg_iter: int = 0, verbose=0):
        """

        Parameters
        ----------
        algorithm: the fixed level optimization algorithm, requires a fit method and score_ and labels_ attributes.
        tol: the minimum modularity increase to keep aggregating.
        max_agg_iter: the maximum number of aggregations to perform
        verbose: enables verbosity
        """
        self.algorithm = algorithm
        self.tol = tol
        if max_agg_iter < 0 or type(max_agg_iter) != int:
            raise ValueError(
                "The maximum number of aggregations should be a positive integer. By default (0), no limit is set.")
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
            raise TypeError('The adjacency matrix should be in a scipy compressed sparse row (csr) format.')
        # check that the graph is not directed
        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError('The adjacency matrix should be square.')
        if (adj_matrix != adj_matrix.T).nnz != 0:
            raise ValueError('The graph should not be directed. Please fit a symmetric adjacency matrix.')
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
                agg_membership = agg_membership[:, sorted(set(agg_membership.nonzero()[1]))]
                membership = membership.dot(agg_membership)
                graph.aggregate(agg_membership)

                if graph.n_nodes == 1:
                    break
            if self.verbose:
                print("Iteration", iteration_count, "completed with", graph.n_nodes, "clusters")
            if self.max_agg_iter != 0:
                if iteration_count == self.max_agg_iter:
                    break

        self.iteration_count_ = iteration_count
        self.labels_ = np.squeeze(np.asarray(membership.argmax(axis=1)))
        index_mapping: dict = {cluster: index for index, cluster in enumerate(set(self.labels_.tolist()))}
        self.labels_ = np.array([index_mapping[cluster] for cluster in self.labels_])
        return self
