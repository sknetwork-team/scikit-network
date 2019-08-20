#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
"""

from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.hierarchy.paris import AggregateGraph
from sknetwork.utils.adjacency_formats import set_adjacency_weights
from sknetwork.utils.checks import check_format


def dasgupta_cost(adjacency: sparse.csr_matrix, dendrogram: np.ndarray, weights: Union[str, np.ndarray] = 'uniform',
                  secondary_weights: Union[None, str, np.ndarray] = None, force_undirected: bool = False,
                  force_biadjacency: bool = False) -> float:
    """
    Dasgupta's cost of a hierarchy (cost metric).

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    dendrogram :
        Dendrogram.
    weights :
        Weights of nodes.
        ``'degree'``, ``'uniform'`` (default) or custom weights.
    secondary_weights :
        Weights of secondary nodes (for bipartite graphs).
        ``None`` (default), ``'degree'``, ``'uniform'`` or custom weights.
        If ``None``, taken equal to weights.
    force_undirected :
        If ``True``, consider the graph as undirected.
    force_biadjacency :
        If ``True``, force the input matrix to be considered as a biadjacency matrix.

    Returns
    -------
    cost : float
        Dasgupta's cost of the hierarchy, normalized to get a value between 0 and 1.

    References
    ----------
    Dasgupta, S. (2016). A cost function for similarity-based hierarchical clustering.
    Proceedings of ACM symposium on Theory of Computing.

    """
    adjacency = check_format(adjacency)
    adjacency, out_weights, in_weights = set_adjacency_weights(adjacency, weights, secondary_weights,
                                                               force_undirected, force_biadjacency)
    n = adjacency.shape[0]
    if n <= 1:
        raise ValueError('The graph must contain at least two nodes.')

    aggregate_graph = AggregateGraph(adjacency + adjacency.T, out_weights, in_weights)

    height = np.zeros(n - 1)
    edge_sampling = np.zeros(n - 1)
    cluster_weight = np.zeros(n - 1)
    for t in range(n - 1):
        node1 = int(dendrogram[t][0])
        node2 = int(dendrogram[t][1])
        if node1 >= n and height[node1 - n] == dendrogram[t][2]:
            edge_sampling[t] = edge_sampling[node1 - n]
            edge_sampling[node1 - n] = 0
        elif node2 >= n and height[node2 - n] == dendrogram[t][2]:
            edge_sampling[t] = edge_sampling[node2 - n]
            edge_sampling[node2 - n] = 0
        height[t] = dendrogram[t][2]
        if node2 in aggregate_graph.neighbors[node1]:
            edge_sampling[t] += aggregate_graph.neighbors[node1][node2]
        cluster_weight[t] = aggregate_graph.cluster_out_weights[node1] + aggregate_graph.cluster_out_weights[node2] \
            + aggregate_graph.cluster_in_weights[node1] + aggregate_graph.cluster_in_weights[node2]
        aggregate_graph.merge(node1, node2)

    cost: float = edge_sampling.dot(cluster_weight) / 2
    return cost


def tree_sampling_divergence(adjacency: sparse.csr_matrix, dendrogram: np.ndarray,
                             weights: Union[str, np.ndarray] = 'degree',
                             secondary_weights: Union[None, str, np.ndarray] = None, force_undirected: bool = False,
                             force_biadjacency: bool = False, normalized: bool = True) -> float:
    """
    Tree sampling divergence of a hierarchy (quality metric).

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    dendrogram :
        Dendrogram.
    weights :
        Weights of nodes.
        ``'degree'`` (default), ``'uniform'`` or custom weights.
    secondary_weights :
        Weights of secondary nodes (for bipartite graphs).
        ``None`` (default), ``'degree'``, ``'uniform'`` or custom weights.
        If ``None``, taken equal to weights.
    force_undirected :
        If ``True``, consider the graph as undirected.
    force_biadjacency :
        If ``True``, force the input matrix to be considered as a biadjacency matrix.
    normalized:
        If ``True``, normalized by the mutual information of the graph.

    Returns
    -------
    quality : float
        The tree sampling divergence of the hierarchy.
        If normalized, return a value between 0 and 1.

    References
    ----------
    Charpentier, B. & Bonald, T. (2019).  Tree Sampling Divergence: An Information-Theoretic Metric for
    Hierarchical Graph Clustering. Proceedings of IJCAI.
    https://hal.telecom-paristech.fr/hal-02144394/document
    """
    adjacency = check_format(adjacency)
    adjacency, out_weights, in_weights = set_adjacency_weights(adjacency, weights, secondary_weights,
                                                               force_undirected, force_biadjacency)
    n = adjacency.shape[0]
    if n <= 1:
        raise ValueError('The graph must contain at least two nodes.')
    total_weight = adjacency.data.sum()
    if total_weight <= 0:
        raise ValueError('The graph must contain at least one edge.')

    adjacency = adjacency / total_weight
    aggregate_graph = AggregateGraph(adjacency + adjacency.T, out_weights, in_weights)

    height = np.zeros(n - 1)
    edge_sampling = np.zeros(n - 1)
    node_sampling = np.zeros(n - 1)
    for t in range(n - 1):
        node1 = int(dendrogram[t][0])
        node2 = int(dendrogram[t][1])
        if node1 >= n and height[node1 - n] == dendrogram[t][2]:
            edge_sampling[t] = edge_sampling[node1 - n]
            edge_sampling[node1 - n] = 0
            node_sampling[t] = node_sampling[node1 - n]
        elif node2 >= n and height[node2 - n] == dendrogram[t][2]:
            edge_sampling[t] = edge_sampling[node2 - n]
            edge_sampling[node2 - n] = 0
            node_sampling[t] = node_sampling[node2 - n]
        if node2 in aggregate_graph.neighbors[node1]:
            edge_sampling[t] += aggregate_graph.neighbors[node1][node2]
        node_sampling[t] += aggregate_graph.cluster_out_weights[node1] * aggregate_graph.cluster_in_weights[node2] + \
            aggregate_graph.cluster_out_weights[node2] * aggregate_graph.cluster_in_weights[node1]
        height[t] = dendrogram[t][2]
        aggregate_graph.merge(node1, node2)

    index = np.where(edge_sampling)[0]
    quality = edge_sampling[index].dot(np.log(edge_sampling[index] / node_sampling[index]))
    if normalized:
        inv_out_weights = sparse.diags(out_weights, shape=(n, n), format='csr')
        inv_out_weights.data = 1 / inv_out_weights.data
        inv_in_weights = sparse.diags(in_weights, shape=(n, n), format='csr')
        inv_in_weights.data = 1 / inv_in_weights.data
        sampling_ratio = inv_out_weights.dot(adjacency.dot(inv_in_weights))
        inv_out_weights.data = np.ones(len(inv_out_weights.data))
        inv_in_weights.data = np.ones(len(inv_in_weights.data))
        edge_sampling = inv_out_weights.dot(adjacency.dot(inv_in_weights))
        mutual_information = edge_sampling.data.dot(np.log(sampling_ratio.data))
        quality /= mutual_information
    return quality
