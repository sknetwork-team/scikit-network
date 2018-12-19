#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu July 10 2018
@author: Nathan de Lara <ndelara@enst.fr>
"""

import numpy as np

from itertools import combinations
from scipy import errstate, sqrt, isinf, sparse


def modularity(adjacency_matrix, partition, resolution=1.0) -> float:
    """
    Compute the modularity of a node partition.
    Parameters
    ----------
    partition: dict or np.ndarray
       The partition of the nodes.
       The keys of the dictionary correspond to the nodes and the values to the communities.
    adjacency_matrix: scipy.csr_matrix or np.ndarray
        The adjacency matrix of the graph (sparse or dense).
    resolution: double, optional
        The resolution parameter in the modularity function (default=1.).

    Returns
    -------
    modularity : float
       The modularity.
    """

    if type(adjacency_matrix) == sparse.csr_matrix:
        adj_matrix = adjacency_matrix
    elif sparse.isspmatrix(adjacency_matrix) or type(adjacency_matrix) == np.ndarray:
        adj_matrix = sparse.csr_matrix(adjacency_matrix)
    else:
        raise TypeError(
            "The argument must be a NumPy array or a SciPy Sparse matrix.")

    n_nodes = adj_matrix.shape[0]
    norm_adj = adj_matrix / adj_matrix.data.sum()
    probs = norm_adj.dot(np.ones(n_nodes))

    if type(partition) == dict:
        labels = np.array([partition[i] for i in range(n_nodes)])
    elif type(partition) == np.ndarray:
        labels = partition.copy()
    else:
        raise TypeError('The partition must be a dictionary or a NumPy array.')

    row = np.arange(n_nodes)
    col = labels
    data = np.ones(n_nodes)
    membership = sparse.csr_matrix((data, (row, col)))

    pos = ((membership.multiply(norm_adj.dot(membership))).dot(np.ones(membership.shape[1]))).sum()
    neg = np.linalg.norm(membership.T.dot(probs)) ** 2
    return float(pos - resolution * neg)


def cocitation_modularity(adjacency_matrix, partition, resolution=1.0) -> float:
    """
    Compute the modularity of a node partition on the normalized cocitation graph
    associated to a network without explicit computation of the cocitation graph.
    Parameters
    ----------
    partition: dict or np.ndarray
       The partition of the nodes.
       The keys of the dictionary correspond to the nodes and the values to the communities.
    adjacency_matrix: scipy.csr_matrix or np.ndarray
        The adjacency matrix of the graph (sparse or dense).
    resolution: double, optional
        The resolution parameter in the modularity function (default=1.).

    Returns
    -------
    modularity : float
       The modularity.
    """

    if type(adjacency_matrix) == sparse.csr_matrix:
        adj_matrix = adjacency_matrix
    elif sparse.isspmatrix(adjacency_matrix) or type(adjacency_matrix) == np.ndarray:
        adj_matrix = sparse.csr_matrix(adjacency_matrix)
    else:
        raise TypeError(
            "The argument must be a NumPy array or a SciPy Sparse matrix.")

    n_nodes = adj_matrix.shape[0]
    out_degree = adj_matrix.dot(np.ones(adj_matrix.shape[0]))
    in_degree = adj_matrix.T.dot(np.ones(adj_matrix.shape[1]))
    total_weight = out_degree.sum()

    with errstate(divide='ignore'):
        in_degree_sqrt = 1.0 / sqrt(in_degree)
    in_degree_sqrt[isinf(in_degree_sqrt)] = 0
    in_degree_sqrt = sparse.spdiags(in_degree_sqrt, [0], adj_matrix.shape[1], adj_matrix.shape[1], format='csr')
    normalized_adjacency = (adj_matrix.dot(in_degree_sqrt)).T

    if type(partition) == dict:
        labels = np.array([partition[i] for i in range(n_nodes)])
    elif type(partition) == np.ndarray:
        labels = partition.copy()
    else:
        raise TypeError('The partition must be a dictionary or a NumPy array.')

    clusters_indices = set(labels.tolist())
    mod = 0.
    for cluster in clusters_indices:
        indicator_vector = (labels == cluster)
        mod += np.linalg.norm(normalized_adjacency.dot(indicator_vector)) ** 2
        mod -= (resolution / total_weight) * (np.dot(out_degree, indicator_vector)) ** 2

    return float(mod / total_weight)


def performance(adjacency_matrix: sparse.csr_matrix, labels: np.ndarray) -> float:
    """
    The performance is the ratio of the total number of intra-cluster edges plus the total number of inter-cluster
    non-edges with the number of potential edges in the graph.
    Parameters
    ----------
    adjacency_matrix: the adjacency matrix of the graph
    labels: the cluster indices, labels[node] = index of the cluster of node.

    Returns
    -------
    the performance metric
    """
    n_nodes, m_nodes = adjacency_matrix.shape
    if n_nodes != m_nodes:
        raise ValueError('The adjacency matrix is not square.')
    bool_adj = abs(adjacency_matrix.sign())

    clusters = set(labels.tolist())
    cluster_indicator = {cluster: (labels == cluster) for cluster in clusters}
    cluster_sizes = {cluster: cluster_indicator[cluster].sum() for cluster in clusters}

    perf = 0.
    for cluster, indicator in cluster_indicator.items():
        perf += indicator.dot(bool_adj.dot(indicator))

    for cluster_i, cluster_j in combinations(clusters, 2):
        perf += 2 * cluster_sizes[cluster_i] * cluster_sizes[cluster_j]
        perf -= 2 * cluster_indicator[cluster_i].dot(bool_adj.dot(cluster_indicator[cluster_j]))

    return perf / n_nodes**2


def cocitation_performance(adjacency_matrix: sparse.csr_matrix, labels: np.ndarray) -> float:
    """
    The performance of the clustering on the normalized cocitation graph associated to the provided adjacency
    matrix without explicit computation of the graph.
    Parameters
    ----------
    adjacency_matrix: the adjacency matrix of the graph
    labels: the cluster indices, labels[node] = index of the cluster of node.

    Returns
    -------
    The performance metric.
    """
    n_nodes, m_nodes = adjacency_matrix.shape
    if n_nodes != m_nodes:
        raise ValueError('The adjacency matrix is not square.')
    bool_adj = abs(adjacency_matrix.sign())

    clusters = set(labels.tolist())
    cluster_indicator = {cluster: (labels == cluster) for cluster in clusters}
    cluster_sizes = {cluster: cluster_indicator[cluster].sum() for cluster in clusters}

    din = bool_adj.T.dot(np.ones(m_nodes))
    with errstate(divide='ignore'):
        din_sqrt = 1.0 / sqrt(din)
    din_sqrt[isinf(din_sqrt)] = 0
    # pseudo inverse square-root in-degree matrix
    dhin = sparse.spdiags(din_sqrt, [0], m_nodes, m_nodes, format='csr')

    norm_backward_adj = (dhin.dot(bool_adj.T)).tocsr()

    perf = 0.
    for cluster, indicator in cluster_indicator.items():
        perf += indicator.dot(bool_adj.dot(norm_backward_adj.dot(indicator)))

    for cluster_i, cluster_j in combinations(clusters, 2):
        perf += 2 * cluster_sizes[cluster_i] * cluster_sizes[cluster_j]
        perf -= 2 * cluster_indicator[cluster_i].dot(
            bool_adj.dot(norm_backward_adj.dot(cluster_indicator[cluster_j])))

    return perf / n_nodes**2
