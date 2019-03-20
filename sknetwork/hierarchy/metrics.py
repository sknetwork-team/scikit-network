#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
"""

import numpy as np
from scipy import sparse
from sknetwork.hierarchy import AggregateGraph
from typing import Union


def dasgupta_cost(adjacency: sparse.csr_matrix, dendrogram: np.ndarray,
                  node_weights: Union[str, np.ndarray] = 'uniform', normalized: bool = True) -> float:
    """Dasgupta's cost of a hierarchy (cost metric)

     Parameters
     ----------
     adjacency :
        Adjacency matrix of the graph.
     dendrogram :
        Each row contains the two merged nodes, the height in the dendrogram, and the size of the corresponding cluster
     node_weights :
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
     S. Dasgupta. A cost function for similarity-based hierarchical clustering.
     In Proceedings of ACM symposium on Theory of Computing, 2016.

    """

    if type(adjacency) != sparse.csr_matrix:
        raise TypeError('The adjacency matrix must be in a scipy compressed sparse row (csr) format.')
    # check that the graph is not directed
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

    aggregate_graph = AggregateGraph(adjacency, node_weights_vec)

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
        cost *= node_weights_vec.sum()
    return cost


def tree_sampling_divergence(adj_matrix: sparse.csr_matrix, dendrogram: np.ndarray,
                             node_weights: Union[str, np.ndarray] = 'degree', normalized: bool = True) -> float:
    """Tree sampling divergence of a hierarchy (quality metric)

     Parameters
     ----------
     adj_matrix :
        Adjacency matrix of the graph.
     dendrogram :
        Each row contains the two merged nodes, the height in the dendrogram, and the size of the corresponding cluster
     node_weights :
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
     T. Bonald, B. Charpentier (2018), Learning Graph Representations by Dendrograms, https://arxiv.org/abs/1807.05087

    """

    if type(adj_matrix) != sparse.csr_matrix:
        raise TypeError('The adjacency matrix must be in a scipy compressed sparse row (csr) format.')
    # check that the graph is not directed
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError('The adjacency matrix must be square.')
    if (adj_matrix != adj_matrix.T).nnz != 0:
        raise ValueError('The graph cannot be directed. Please fit a symmetric adjacency matrix.')

    n_nodes = adj_matrix.shape[0]

    if type(node_weights) == np.ndarray:
        if len(node_weights) != n_nodes:
            raise ValueError('The number of node weights must match the number of nodes.')
        else:
            node_weights_vec = node_weights
    elif type(node_weights) == str:
        if node_weights == 'degree':
            node_weights_vec = adj_matrix.dot(np.ones(n_nodes))
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

    aggregate_graph = AggregateGraph(adj_matrix, node_weights_vec)

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
        inv_node_weights = sparse.diags(1 / node_weights_vec, shape=(n_nodes, n_nodes), format='csr')
        sampling_ratio = inv_node_weights.dot(adj_matrix.dot(inv_node_weights)) / adj_matrix.data.sum()
        mutual_information = np.sum(adj_matrix.data / adj_matrix.data.sum() * np.log(2 * sampling_ratio.data))
        quality /= mutual_information
    return quality
