#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
"""

import numpy as np
from scipy import sparse

from sknetwork.hierarchy.paris import AggregateGraph
from sknetwork.utils.checks import check_format, check_probs, is_square


# noinspection DuplicatedCode
def dasgupta_score(adjacency: sparse.csr_matrix, dendrogram: np.ndarray, weights: str = 'uniform') -> float:
    """
    Dasgupta's score of a hierarchy, defined as 1 - Dasgupta's cost.

    The higher the score, the better.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    dendrogram :
        Dendrogram.
    weights :
        Weights of nodes.
        ``'degree'`` or ``'uniform'`` (default).

    Returns
    -------
    score : float
        Dasgupta's score of the hierarchy, normalized to get a value between 0 and 1.

    References
    ----------
    Dasgupta, S. (2016). A cost function for similarity-based hierarchical clustering.
    Proceedings of ACM symposium on Theory of Computing.

    """
    adjacency = check_format(adjacency)
    if not is_square(adjacency):
        raise ValueError('The adjacency matrix is not square.')

    n = adjacency.shape[0]
    if n <= 1:
        raise ValueError('The graph must contain at least two nodes.')

    out_weights = check_probs(weights, adjacency)
    in_weights = check_probs(weights, adjacency.T)

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
    return 1 - cost


# noinspection DuplicatedCode
def tree_sampling_divergence(adjacency: sparse.csr_matrix, dendrogram: np.ndarray, weights: str = 'degree',
                             normalized: bool = True) -> float:
    """
    Tree sampling divergence of a hierarchy (quality metric).

    The higher the score, the better.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    dendrogram :
        Dendrogram.
    weights :
        Weights of nodes.
        ``'degree'`` (default) or ``'uniform'``.
    normalized:
        If ``True``, normalized by the mutual information of the graph.

    Returns
    -------
    score : float
        The tree sampling divergence of the hierarchy.
        If normalized, returns a value between 0 and 1.

    References
    ----------
    Charpentier, B. & Bonald, T. (2019).
    `Tree Sampling Divergence: An Information-Theoretic Metric for
    Hierarchical Graph Clustering.
    <https://hal.telecom-paristech.fr/hal-02144394/document>`_
    Proceedings of IJCAI.

    """
    adjacency = check_format(adjacency)
    if not is_square(adjacency):
        raise ValueError('The adjacency matrix is not square.')

    n = adjacency.shape[0]
    if n <= 1:
        raise ValueError('The graph must contain at least two nodes.')

    total_weight = adjacency.data.sum()
    if total_weight <= 0:
        raise ValueError('The graph must contain at least one edge.')

    adjacency.data = adjacency.data / total_weight

    out_weights = check_probs(weights, adjacency)
    in_weights = check_probs(weights, adjacency.T)

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
    score = edge_sampling[index].dot(np.log(edge_sampling[index] / node_sampling[index]))
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
        score /= mutual_information
    return score
