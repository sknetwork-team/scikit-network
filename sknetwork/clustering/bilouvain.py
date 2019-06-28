#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 3, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

from sknetwork.clustering.louvain import *
from sknetwork.utils.adjacency_formats import bipartite2undirected
from sknetwork.utils.checks import *
from sknetwork.utils.algorithm_base_class import Algorithm
from sknetwork import njit, prange


@njit(parallel=False)
def fit_core(n_samp, node_probs, indptr, indices, data, resolution, tol):
    n_feat = len(node_probs) - n_samp
    increase = True
    total_increase = 0.

    labels: np.ndarray = np.arange(n_samp + n_feat)
    samp_clusters_probs: np.ndarray = np.hstack((node_probs[:n_samp], np.zeros(n_feat)))
    feat_clusters_probs: np.ndarray = np.hstack((np.zeros(n_samp), node_probs[n_samp:]))

    while increase:
        increase = False
        pass_increase = 0.

        for nodes, source_clusters_probs, target_clusters_probs in ((prange(n_samp), samp_clusters_probs,
                                                                     feat_clusters_probs),
                                                                    (prange(n_samp, n_samp + n_feat),
                                                                     feat_clusters_probs, samp_clusters_probs)):

            increases, labels, source_clusters_probs = local_updates(nodes, node_probs, indptr, indices, data, labels,
                                                                     source_clusters_probs, target_clusters_probs,
                                                                     resolution)
            pass_increase += increases.sum()
        total_increase += pass_increase
        if pass_increase > tol:
            increase = True

    return labels, total_increase


@njit(parallel=True)
def local_updates(nodes, node_probs, indptr, indices, data, labels, source_clusters_probs, target_clusters_probs,
                  resolution):
    increases = np.zeros(len(node_probs))
    for node in nodes:
        node_cluster: int = labels[node]
        unique_clusters = set(labels[indices[indptr[node]:indptr[node + 1]]])
        unique_clusters.discard(node_cluster)
        n_clusters = len(unique_clusters)

        if n_clusters > 0:
            node_proba: float = node_probs[node]
            out_delta: float = resolution * node_proba * target_clusters_probs[node_cluster]

            unique_clusters_list = list(unique_clusters)
            neighbor_cluster_weights = np.full(n_clusters, 0.0)

            for ix in range(indptr[node], indptr[node + 1]):
                neighbor_cluster = labels[indices[ix]]
                if neighbor_cluster == node_cluster:
                    out_delta -= data[ix]
                else:
                    neighbor_cluster_ix = unique_clusters_list.index(neighbor_cluster)
                    neighbor_cluster_weights[neighbor_cluster_ix] += data[ix]

            best_delta = 0.0
            best_cluster = node_cluster

            for ix, cluster in enumerate(unique_clusters):
                in_delta: float = neighbor_cluster_weights[ix]
                in_delta -= resolution * node_proba * target_clusters_probs[cluster]

                local_delta = out_delta + in_delta
                if local_delta > best_delta:
                    best_delta = local_delta
                    best_cluster = cluster

            if best_delta > 0:
                increases[node] = best_delta
                source_clusters_probs[node_cluster] -= node_proba
                source_clusters_probs[best_cluster] += node_proba
                labels[node] = best_cluster

    return increases, labels, source_clusters_probs


class BiLouvain(Algorithm):
    """
    BiLouvain algorithm for graph clustering in Python (default) and Numba.

    Parameters
    ----------

    resolution:
        Resolution parameter.
    tol:
        Minimum increase in the objective function to enter a new optimization pass.
    agg_tol:
        Minimum increase in the objective function to enter a new aggregation pass.
    max_agg_iter:
        Maximum number of aggregations.
        A negative value is interpreted as no limit.
    engine: str
        ``'default'``, ``'python'`` or ``'numba'``. If ``'default'``, it will tests if numba is available.
    verbose:
        Verbose mode.

    Attributes
    ----------
    labels_: np.ndarray
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

    def __init__(self, resolution: float = 1, tol: float = 1e-3, agg_tol: float = 1e-3, max_agg_iter: int = -1,
                 engine='default', verbose: bool = False):
        self.resolution = resolution
        self.tol = tol
        self.agg_tol = agg_tol
        if type(max_agg_iter) != int:
            raise TypeError('The maximum number of iterations must be an integer.')
        self.max_agg_iter = max_agg_iter
        self.engine = check_engine(engine)
        self.verbose = verbose
        self.labels_ = None
        self.feature_labels_ = None
        self.iteration_count_ = None
        self.aggregate_graph_ = None
        self.score_ = None
        self.n_clusters_ = None

    def fit(self, biadjacency: sparse.csr_matrix, weights: Union['str', np.ndarray] = 'degree',
            feature_weights: Union['str', np.ndarray] = 'degree'):
        """Alternates local optimization and aggregation until convergence.

        Parameters
        ----------
        biadjacency:
            adjacency matrix of the graph to cluster, treated as a biadjacency matrix
        weights:
            Probabilities for the samples in the null model. ``'degree'``, ``'uniform'`` or custom weights.
        feature_weights:
            Probabilities for the features in the null model. ``'degree'``, ``'uniform'`` or custom weights.

        Returns
        -------
        self: :class:`BiLouvain`
        """
        biadjacency = check_format(biadjacency)
        n_samp, n_feat = biadjacency.shape

        samp_weights = np.hstack((check_probs(weights, biadjacency), np.zeros(n_feat)))
        feat_weights = np.hstack((np.zeros(n_samp), check_probs(feature_weights, biadjacency.T)))
        graph = NormalizedGraph(bipartite2undirected(biadjacency), samp_weights, feat_weights)

        iteration_count: int = 0
        if self.verbose:
            print("Starting with", biadjacency.shape, "nodes")

        labels, total_increase = fit_core(n_samp, graph.node_probs, graph.norm_adjacency.indptr,
                                          graph.norm_adjacency.indices, graph.norm_adjacency.data, self.resolution,
                                          self.tol)
        _, labels = np.unique(labels, return_inverse=True)
        iteration_count += 1

        membership = membership_matrix(labels)
        graph.aggregate(membership)
        if self.verbose:
            print("Initial iteration completed with", graph.norm_adjacency.shape, "clusters")

        louvain = Louvain(GreedyModularity(self.resolution, self.tol, engine=self.engine), verbose=self.verbose)
        louvain.fit(graph.norm_adjacency)
        iteration_count += louvain.iteration_count_

        membership = membership.dot(membership_matrix(louvain.labels_))

        self.n_clusters_ = louvain.n_clusters_
        self.iteration_count_ = iteration_count
        self.labels_ = membership.indices[:n_samp]
        self.feature_labels_ = membership.indices[n_samp:]
        self.aggregate_graph_ = louvain.aggregate_graph_ * biadjacency.data.sum()
        return self
