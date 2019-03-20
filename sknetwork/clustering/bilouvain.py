#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 3, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

try:
    from numba import njit, prange
    default = 'numba'
except ImportError:
    def njit(func):
        return func
    prange = range
    default = 'python'

import numpy as np
from scipy import sparse
from typing import Union


class BipartiteGraph:
    """
    A class of graphs suitable for the BiLouvain algorithm.

    Attributes
    ----------
    n_samples: int,
        the cardinal of V1 (the set of samples)
    n_features: int,
        the cardinal of V2 (the set of features)
    sample_weights: np.ndarray,
        the normalized degree vector of the samples
    feature_weights: np.ndarray,
        the normalized degree vector of the features

    """

    def __init__(self, biadjacency: sparse.csr_matrix):
        """

        Parameters
        ----------
        biadjacency: biadjacency matrix of the graph as SciPy sparse matrix
        """
        self.n_samples: int = biadjacency.shape[0]
        self.n_features: int = biadjacency.shape[1]
        self.norm_adjacency: sparse.csr_matrix = biadjacency / biadjacency.data.sum()
        self.sample_weights: np.ndarray = self.norm_adjacency.dot(np.ones(self.n_features))
        self.feature_weights: np.ndarray = self.norm_adjacency.T.dot(np.ones(self.n_samples))

    def aggregate(self, sample_membership: sparse.csr_matrix, feature_membership: sparse.csr_matrix):
        """
        Aggregates nodes belonging to the same clusters while keeping the bipartite structure.

        Parameters
        ----------
        sample_membership:
            matrix of shape n_samples x n_clusters,
            row number i is the one-hot cluster vector of sample i
        feature_membership:
            matrix of shape n_features x n_clusters,
            row number i is the one-hot cluster vector of feature i

        Returns
        -------
        self: :class:`BipartiteGraph`
        """
        self.norm_adjacency = sample_membership.T.dot(self.norm_adjacency.dot(feature_membership)).tocsr()
        self.sample_weights = np.array(sample_membership.T.dot(self.sample_weights).T)
        self.feature_weights = np.array(feature_membership.T.dot(self.feature_weights).T)
        self.n_samples, self.n_features = self.norm_adjacency.shape
        return self


class BiOptimizer:
    """
    A generic optimization algorithm.

    Attributes
    ----------
    score_: float
        Total increase of the objective function.
    sample_labels_: np.ndarray
        Cluster index of each node in V1.
    feature_labels_: np.ndarray
        Cluster index of each node in V2.
    """
    def __init__(self):
        self.score_ = None
        self.sample_labels_ = None
        self.feature_labels_ = None

    def fit(self, graph: BipartiteGraph):
        """Fit the clusters to the objective function.

         Parameters
         ----------
         graph:
             Graph to cluster.

         Returns
         -------
         self: :class:`BiOptimizer`

         """
        return self


class GreedyBipartite(BiOptimizer):
    """
    A greedy bipartite modularity optimizer.

    Attributes
    ----------
    resolution:
        modularity resolution
    tol:
        minimum bimodularity increase to enter a new optimization pass
    """

    def __init__(self, resolution: float = 1, tol: float = 1e-3):
        BiOptimizer.__init__(self)
        self.resolution = resolution
        self.tol = tol

    def fit(self, graph: BipartiteGraph):
        """Iterates over the nodes of the graph and moves them to the cluster of highest increase among their neighbors.

        Parameters
        ----------
        graph:
            the graph to cluster

        Returns
        -------
        self: :class:`BiOptimizer`
        """
        increase: bool = True
        total_increase: float = 0.

        sample_labels: np.ndarray = np.arange(graph.n_samples)
        feature_labels: np.ndarray = graph.n_samples + np.arange(graph.n_features)
        sample_clusters_proba: np.ndarray = np.hstack((graph.sample_weights.copy(), np.zeros(graph.n_features)))
        feature_clusters_proba: np.ndarray = np.hstack((np.zeros(graph.n_samples), graph.feature_weights.copy()))

        sample_indptr: np.ndarray = graph.norm_adjacency.indptr
        sample_indices: np.ndarray = graph.norm_adjacency.indices
        sample_data: np.ndarray = graph.norm_adjacency.data

        transposed_adjacency = graph.norm_adjacency.T.tocsr()
        feature_indptr: np.ndarray = transposed_adjacency.indptr
        feature_indices: np.ndarray = transposed_adjacency.indices
        feature_data: np.ndarray = transposed_adjacency.data
        del transposed_adjacency

        while increase:
            increase = False
            pass_increase: float = 0.

            sample_args = (graph.n_samples, graph.sample_weights, sample_labels, sample_clusters_proba,
                           sample_indptr, sample_indices, sample_data)
            feature_args = (graph.n_features, graph.feature_weights, feature_labels, feature_clusters_proba,
                            feature_indptr, feature_indices, feature_data)

            for source, target in [(sample_args, feature_args), (feature_args, sample_args)]:
                n_nodes, source_weights, source_labels, source_cluster_proba, indptr, indices, data = source
                target_labels, target_cluster_proba = target[2], target[3]

                for node in range(n_nodes):
                    node_cluster: int = source_labels[node]
                    target_weights: np.ndarray = data[indptr[node]:indptr[node + 1]]
                    neighbors: np.ndarray = indices[indptr[node]:indptr[node + 1]]
                    neighbors_clusters: np.ndarray = target_labels[neighbors]
                    unique_clusters: list = list(set(neighbors_clusters.tolist()) - {node_cluster})
                    n_clusters: int = len(unique_clusters)

                    if n_clusters > 0:
                        node_proba: float = source_weights[node]
                        node_ratio: float = self.resolution * node_proba

                        # total weight of edges to neighbors in the same cluster
                        out_delta: float = target_weights[neighbors_clusters == node_cluster].sum()
                        # minus the probability to choose a neighbor in the same cluster (with resolution factor)
                        out_delta -= node_ratio * target_cluster_proba[node_cluster]

                        local_delta: np.ndarray = np.full(n_clusters, -out_delta)

                        for index_cluster, cluster in enumerate(unique_clusters):
                            # total weight of edges to neighbors in the candidate cluster
                            in_delta: float = target_weights[neighbors_clusters == cluster].sum()
                            # minus the probability to choose a neighbor in the candidate cluster
                            in_delta -= node_ratio * target_cluster_proba[cluster]

                            local_delta[index_cluster] += in_delta

                        delta_argmax: int = local_delta.argmax()
                        best_delta: float = local_delta[delta_argmax]
                        if best_delta > 0:
                            pass_increase += best_delta
                            best_cluster = unique_clusters[delta_argmax]

                            source_cluster_proba[node_cluster] -= node_proba
                            source_cluster_proba[best_cluster] += node_proba
                            source_labels[node] = best_cluster

            total_increase += pass_increase
            if pass_increase > self.tol:
                increase = True

        self.score_ = total_increase
        _, self.sample_labels_ = np.unique(sample_labels, return_inverse=True)
        _, self.feature_labels_ = np.unique(feature_labels, return_inverse=True)

        return self


@njit
def fit_core(sample_args, feature_args, resolution, tol):
    n_samples, n_features = sample_args[0], feature_args[0]
    sample_weights, feature_weights = sample_args[1], feature_args[1]

    increase = True
    total_increase = 0.

    sample_labels: np.ndarray = np.arange(n_samples)
    feature_labels: np.ndarray = n_samples + np.arange(n_features)

    sample_clusters_proba: np.ndarray = np.hstack((sample_weights.copy(), np.zeros(n_features)))
    feature_clusters_proba: np.ndarray = np.hstack((np.zeros(n_samples), feature_weights.copy()))

    while increase:
        increase = False
        pass_increase = 0.

        all_sample_args = (sample_args, sample_labels, sample_clusters_proba)
        all_feature_args = (feature_args, feature_labels, feature_clusters_proba)

        for source, target in [(all_sample_args, all_feature_args), (all_feature_args, all_sample_args)]:
            source_args, source_labels, source_clusters_proba = source
            target_args, target_labels, target_clusters_proba = target

            n_nodes, source_weights, indptr, indices, data = source_args
            for node in prange(n_nodes):
                node_cluster: int = source_labels[node]
                unique_clusters = set(target_labels[indices[indptr[node]:indptr[node + 1]]])
                unique_clusters.discard(node_cluster)
                n_clusters = len(unique_clusters)

                if n_clusters > 0:
                    node_proba: float = source_weights[node]
                    out_delta: float = resolution * node_proba * target_clusters_proba[node_cluster]

                    unique_clusters_list = list(unique_clusters)
                    neighbor_cluster_weights = np.full(n_clusters, 0.0)

                    for ix in range(indptr[node], indptr[node + 1]):
                        neighbor_cluster = target_labels[indices[ix]]
                        if neighbor_cluster == node_cluster:
                            out_delta -= data[ix]
                        else:
                            neighbor_cluster_ix = unique_clusters_list.index(neighbor_cluster)
                            neighbor_cluster_weights[neighbor_cluster_ix] += data[ix]

                    best_delta = 0.0
                    best_cluster = node_cluster

                    for ix, cluster in enumerate(unique_clusters):
                        in_delta: float = neighbor_cluster_weights[ix]
                        in_delta -= resolution * node_proba * target_clusters_proba[cluster]

                        local_delta = out_delta + in_delta
                        if local_delta > best_delta:
                            best_delta = local_delta
                            best_cluster = cluster

                    if best_delta > 0:
                        pass_increase += best_delta
                        source_clusters_proba[node_cluster] -= node_proba
                        source_clusters_proba[best_cluster] += node_proba
                        source_labels[node] = best_cluster

        total_increase += pass_increase
        if pass_increase > tol:
            increase = True

    return sample_labels, feature_labels, total_increase


class GreedyBipartiteNumba(BiOptimizer):
    """
    A greedy modularity optimizer using Numba for enhanced performance.

    Attributes
    ----------
    resolution:
        bimodularity resolution
    tol:
        minimum bimodularity increase to enter a new optimization pass

    Notes
    -----
    Tested with Numba v0.42.0.
    """

    def __init__(self, resolution: float = 1., tol: float = 1e-3):
        BiOptimizer.__init__(self)
        self.resolution = resolution
        self.tol = tol

    def fit(self, graph: BipartiteGraph):
        """Iterates over the nodes of the graph and moves them to the cluster of highest increase among their neighbors.

        Parameters
        ----------
        graph:
            the graph to cluster

        Returns
        -------
        self: :class:`BiOptimizer`

        """
        transposed_adjacency = graph.norm_adjacency.T.tocsr()
        sample_args = (graph.n_samples, graph.sample_weights,
                       graph.norm_adjacency.indptr, graph.norm_adjacency.indices, graph.norm_adjacency.data)
        feature_args = (graph.n_features, graph.feature_weights,
                        transposed_adjacency.indptr, transposed_adjacency.indices, transposed_adjacency.data)

        sample_labels, feature_labels, total_increase = fit_core(sample_args, feature_args,
                                                                 self.resolution, self.tol)

        self.score_ = total_increase
        _, self.sample_labels_ = np.unique(sample_labels, return_inverse=True)
        _, self.feature_labels_ = np.unique(feature_labels, return_inverse=True)

        return self


class BiLouvain:
    """
    BiLouvain algorithm for graph clustering in Python (default) and Numba.

    Attributes
    ----------
    sample_labels_: np.ndarray
        Cluster index of each node in V1.
    feature_labels_: np.ndarray
        Cluster index of each node in V2.
    iteration_count_: int
        Total number of aggregations performed.
    aggregate_graph_: sparse.csr_matrix
        Aggregated graph at the end of the algorithm.
    score_: float
        objective function value after fit
    n_clusters_: int
        number of clusters after fit
    """

    def __init__(self, algorithm: Union[str, BiOptimizer] = default, resolution: float = 1, tol: float = 1e-3,
                 agg_tol: float = 1e-3, max_agg_iter: int = -1, verbose: bool = False):
        """
        Parameters
        ----------
        algorithm:
            The optimization algorithm.
            Requires a fit method.
            Requires score_, sample_labels_,  and labels_ attributes.
        resolution:
            Resolution parameter.
        tol:
            Minimum increase in the objective function to enter a new optimization pass.
        agg_tol:
            Minimum increase in the objective function to enter a new aggregation pass.
        max_agg_iter:
            Maximum number of aggregations.
            A negative value is interpreted as no limit.
        verbose:
            Verbose mode.
        """
        if type(algorithm) == str:
            if algorithm == "numba":
                self.algorithm = GreedyBipartiteNumba(resolution, tol)
            elif algorithm == "python":
                self.algorithm = GreedyBipartite(resolution, tol)
            else:
                raise ValueError('Unknown algorithm name.')
        elif isinstance(algorithm, BiOptimizer):
            self.algorithm = algorithm
        else:
            raise TypeError('Algorithm must be a string ("numba" or "python") or a valid algorithm.')

        if type(max_agg_iter) != int:
            raise TypeError('The maximum number of iterations must be an integer.')
        self.agg_tol = agg_tol
        self.max_agg_iter = max_agg_iter
        self.verbose = verbose
        self.sample_labels_ = None
        self.feature_labels_ = None
        self.iteration_count_ = None
        self.aggregate_graph_ = None
        self.score_ = None
        self.n_clusters_ = None

    def fit(self, biadjacency: sparse.csr_matrix):
        """Alternates local optimization and aggregation until convergence.

        Parameters
        ----------
        biadjacency:
            adjacency matrix of the graph to cluster, treated as a biadjacency matrix

        Returns
        -------
        self: :class:`BiLouvain`
        """
        if type(biadjacency) != sparse.csr_matrix:
            raise TypeError('The adjacency matrix must be in a scipy compressed sparse row (csr) format.')
        graph = BipartiteGraph(biadjacency)
        sample_membership = sparse.identity(graph.n_samples, format='csr')
        feature_membership = sparse.identity(graph.n_features, format='csr')
        increase: bool = True
        iteration_count: int = 0
        if self.verbose:
            print("Starting with", biadjacency.shape, "nodes")

        self.score_ = 0.
        while increase:
            iteration_count += 1
            self.algorithm.fit(graph)
            if self.algorithm.score_ - self.score_ <= self.agg_tol:
                increase = False
            else:
                self.score_ = self.algorithm.score_

                sample_row = np.arange(graph.n_samples)
                sample_col = self.algorithm.sample_labels_
                sample_data = np.ones(graph.n_samples)
                sample_agg_membership = sparse.csr_matrix((sample_data, (sample_row, sample_col)))
                sample_membership = sample_membership.dot(sample_agg_membership)

                feature_row = np.arange(graph.n_features)
                feature_col = self.algorithm.feature_labels_
                feature_data = np.ones(graph.n_features)
                feature_agg_membership = sparse.csr_matrix((feature_data, (feature_row, feature_col)))
                feature_membership = feature_membership.dot(feature_agg_membership)

                graph.aggregate(sample_agg_membership, feature_agg_membership)

            if self.verbose:
                print("Iteration", iteration_count, "completed with",
                      graph.n_samples, graph.n_features, "clusters")
                print(self.algorithm.score_)
            if iteration_count == self.max_agg_iter:
                break

        self.iteration_count_ = iteration_count
        self.sample_labels_ = sample_membership.indices
        _, self.sample_labels_ = np.unique(self.sample_labels_, return_inverse=True)
        self.feature_labels_ = feature_membership.indices
        _, self.feature_labels_ = np.unique(self.feature_labels_, return_inverse=True)
        self.n_clusters_ = max(max(self.sample_labels_), max(self.feature_labels_)) + 1
        self.aggregate_graph_ = graph.norm_adjacency * biadjacency.data.sum()
        return self
