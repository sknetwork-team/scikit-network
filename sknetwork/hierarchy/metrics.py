#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
"""

from sknetwork.hierarchy import AggregateGraph
from sknetwork.utils.checks import *


def dasgupta_cost(adjacency: sparse.csr_matrix, dendrogram: np.ndarray,
                  weights: Union[str, np.ndarray] = 'uniform', normalized: bool = True) -> float:
    """Dasgupta's cost of a hierarchy (cost metric)

     Parameters
     ----------
     adjacency :
        Adjacency matrix of the graph.
     dendrogram :
        Each row contains the two merged nodes, the height in the dendrogram, and the size of the corresponding cluster
     weights :
        Vector of node weights. Default = 'uniform', weight 1 for each node.
     normalized:
        If true, normalized by the number of ndoes of the graph.

     Returns
     -------
     cost : float
         Dasgupta's cost of the hierarchy.
         Normalized by the number of nodes to get a value between 0 and 1.

     References
     ----------
     S. Dasgupta (2016). A cost function for similarity-based hierarchical clustering.
     Proceedings of ACM symposium on Theory of Computing.

    """
    adjacency = check_format(adjacency)

    if not is_square(adjacency):
        raise ValueError('The adjacency matrix must be square.')
    if adjacency.shape[0] <= 1:
        raise ValueError('The graph must contain at least two nodes.')
    if not is_symmetric(adjacency):
        raise ValueError('The graph must be undirected. Please fit a symmetric adjacency matrix.')

    node_probs = check_probs(weights, adjacency, positive_entries=True)

    n_nodes = adjacency.shape[0]

    aggregate_graph = AggregateGraph(adjacency, node_probs)

    height = np.zeros(n_nodes - 1)
    edge_sampling = np.zeros(n_nodes - 1)
    cluster_weight = np.zeros(n_nodes - 1)
    for t in range(n_nodes - 1):
        node1 = int(dendrogram[t][0])
        node2 = int(dendrogram[t][1])
        if node1 >= n_nodes and height[node1 - n_nodes] == dendrogram[t][2]:
            edge_sampling[t] = edge_sampling[node1 - n_nodes]
            edge_sampling[node1 - n_nodes] = 0
        elif node2 >= n_nodes and height[node2 - n_nodes] == dendrogram[t][2]:
            edge_sampling[t] = edge_sampling[node2 - n_nodes]
            edge_sampling[node2 - n_nodes] = 0
        height[t] = dendrogram[t][2]
        edge_sampling[t] += 2 * aggregate_graph.graph[node1][node2]
        cluster_weight[t] = aggregate_graph.cluster_probs[node1] + aggregate_graph.cluster_probs[node2]
        aggregate_graph.merge(node1, node2)

    cost: float = (edge_sampling * cluster_weight).sum()
    if not normalized:
        cost *= node_probs.sum()
    return cost


def tree_sampling_divergence(adjacency: sparse.csr_matrix, dendrogram: np.ndarray,
                             weights: Union[str, np.ndarray] = 'degree', normalized: bool = True) -> float:
    """Tree sampling divergence of a hierarchy (quality metric)

     Parameters
     ----------
     adjacency :
        Adjacency matrix of the graph.
     dendrogram :
        Each row contains the two merged nodes, the height in the dendrogram, and the size of the corresponding cluster
     weights :
        Vector of node weights. Default = 'degree', weight of each node in the graph.
     normalized:
        If true, normalized by the mutual information of the graph.

     Returns
     -------
     quality : float
         The tree sampling divergence of the hierarchy (quality metric).
         Normalized by the mutual information to get a value between 0 and 1.

     References
     ----------
     T. Bonald, B. Charpentier (2018). Learning Graph Representations by Dendrograms.
     https://arxiv.org/abs/1807.05087

    """
    adjacency = check_format(adjacency)

    if not is_square(adjacency):
        raise ValueError('The adjacency matrix must be square.')
    if adjacency.shape[0] <= 1:
        raise ValueError('The graph must contain at least two nodes.')
    if not is_symmetric(adjacency):
        raise ValueError('The graph must be undirected. Please fit a symmetric adjacency matrix.')

    node_probs = check_probs(weights, adjacency, positive_entries=True)

    n_nodes = adjacency.shape[0]

    aggregate_graph = AggregateGraph(adjacency, node_probs)

    height = np.zeros(n_nodes - 1)
    edge_sampling = np.zeros(n_nodes - 1)
    node_sampling = np.zeros(n_nodes - 1)
    for t in range(n_nodes - 1):
        node1 = int(dendrogram[t][0])
        node2 = int(dendrogram[t][1])
        if node1 >= n_nodes and height[node1 - n_nodes] == dendrogram[t][2]:
            edge_sampling[t] = edge_sampling[node1 - n_nodes]
            edge_sampling[node1 - n_nodes] = 0
            node_sampling[t] = node_sampling[node1 - n_nodes]
        elif node2 >= n_nodes and height[node2 - n_nodes] == dendrogram[t][2]:
            edge_sampling[t] = edge_sampling[node2 - n_nodes]
            edge_sampling[node2 - n_nodes] = 0
            node_sampling[t] = node_sampling[node2 - n_nodes]
        edge_sampling[t] += 2 * aggregate_graph.graph[node1][node2]
        node_sampling[t] += aggregate_graph.cluster_probs[node1] * aggregate_graph.cluster_probs[node2]
        height[t] = dendrogram[t][2]
        aggregate_graph.merge(node1, node2)

    index = np.where(edge_sampling)[0]
    quality: float = np.sum(edge_sampling[index] * np.log(edge_sampling[index] / node_sampling[index]))
    if normalized:
        inv_node_weights = sparse.diags(1 / node_probs, shape=(n_nodes, n_nodes), format='csr')
        sampling_ratio = inv_node_weights.dot(adjacency.dot(inv_node_weights)) / adjacency.data.sum()
        mutual_information = np.sum(adjacency.data / adjacency.data.sum() * np.log(2 * sampling_ratio.data))
        quality /= mutual_information
    return quality
