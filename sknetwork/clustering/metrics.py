#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu July 10 2018
@author: Nathan de Lara <ndelara@enst.fr>
"""

import numpy as np

from itertools import combinations
from scipy import sparse
from typing import Union


def bimodularity(biadjacency: sparse.csr_matrix, sample_labels: np.ndarray, feature_labels: np.ndarray,
                 resolution: float = 1) -> float:
    """
    Modularity for bipartite graphs.

    :math:`Q = \\sum_{ij}(\\dfrac{B_{ij}}{w} - \\gamma \\dfrac{d_if_j}{w^2})\\delta_{ij}`

    Parameters
    ----------
    biadjacency:
        matrix of shape n1 x n2
    sample_labels:
        partition of the samples, sample_labels[node] = cluster_index
    feature_labels:
        partition of the features, feature_labels[node] = cluster_index
    resolution:
        scaling for the second term of the bimodularity

    Returns
    -------
    bimodularity: float
        The bimodularity
    """
    n_samples, n_features = biadjacency.shape
    one_samples, one_features = np.ones(n_samples), np.ones(n_features)
    total_weight: float = biadjacency.data.sum()
    sample_weights = biadjacency.dot(one_features) / total_weight
    features_weights = biadjacency.T.dot(one_samples) / total_weight

    _, sample_labels = np.unique(sample_labels, return_inverse=True)
    _, feature_labels = np.unique(feature_labels, return_inverse=True)
    n_clusters = max(len(set(sample_labels)), len(set(feature_labels)))

    sample_membership = sparse.csr_matrix((one_samples, (np.arange(n_samples), sample_labels)),
                                          shape=(n_samples, n_clusters))
    feature_membership = sparse.csc_matrix((one_features, (np.arange(n_features), feature_labels)),
                                           shape=(n_features, n_clusters))

    fit: float = sample_membership.multiply(biadjacency.dot(feature_membership)).data.sum() / total_weight
    div: float = (sample_membership.T.dot(sample_weights)).dot(feature_membership.T.dot(features_weights))

    return fit - resolution * div


def modularity(adjacency: Union[sparse.csr_matrix, np.ndarray], partition: Union[dict, np.ndarray],
               resolution: float = 1, directed: bool = False) -> float:
    """
    Compute the modularity of a node partition.

    :math:`Q = \\sum_{ij}(\\dfrac{A_{ij}}{w} - \\gamma \\dfrac{d_id_j}{w^2})\\delta_{ij}` for undirected graphs,

    :math:`Q = \\sum_{ij}(\\dfrac{A_{ij}}{w} - \\gamma \\dfrac{d_if_j}{w^2})\\delta_{ij}` for directed graphs.

    Parameters
    ----------
    partition : dict or np.ndarray
       The partition of the nodes. The keys of the dictionary correspond to the nodes and the values to the communities.
    adjacency : scipy.csr_matrix or np.ndarray
        The adjacency matrix of the graph (sparse or dense).
    resolution : float, optional (default=1.)
        The resolution parameter in the modularity function.
    directed: bool
        Whether to compute the modularity for directed graphs or not.

    Returns
    -------
    modularity : float
        The modularity.
    """

    if type(adjacency) == sparse.csr_matrix:
        adj_matrix = adjacency
    elif sparse.isspmatrix(adjacency) or type(adjacency) == np.ndarray:
        adj_matrix = sparse.csr_matrix(adjacency)
    else:
        raise TypeError(
            "The argument must be a NumPy array or a SciPy Sparse matrix.")

    n_nodes, m_nodes = adj_matrix.shape
    if n_nodes != m_nodes:
        raise ValueError('The adjacency must be a square matrix.')
    norm_adj = adj_matrix / adj_matrix.data.sum()
    probs = norm_adj.dot(np.ones(n_nodes))

    if type(partition) == dict:
        labels = np.array([partition[i] for i in range(n_nodes)])
    elif type(partition) == np.ndarray:
        labels = partition.copy()
    else:
        raise TypeError('The partition must be a dictionary or a NumPy array.')

    if directed:
        return bimodularity(adjacency, labels, labels, resolution)
    else:
        row = np.arange(n_nodes)
        col = labels
        data = np.ones(n_nodes)
        membership = sparse.csr_matrix((data, (row, col)))

        fit = ((membership.multiply(norm_adj.dot(membership))).dot(np.ones(membership.shape[1]))).sum()
        diversity = np.linalg.norm(membership.T.dot(probs)) ** 2
        return float(fit - resolution * diversity)


def cocitation_modularity(adjacency, partition, resolution: float = 1) -> float:
    """
    Compute the modularity of a node partition on the normalized cocitation graph
    associated to a network without explicit computation of the cocitation graph.

    :math:`Q = \\sum_{ij}(\\dfrac{(AF^{-1}A^T)_{ij}}{w} - \\gamma \\dfrac{d_id_j}{w^2})\\delta_{ij}`

    Parameters
    ----------
    partition: dict or np.ndarray
       The partition of the nodes. The keys of the dictionary correspond to the nodes and the values to the communities.
    adjacency: scipy.csr_matrix or np.ndarray
        The adjacency matrix of the graph (sparse or dense).
    resolution: float (default=1.)
        The resolution parameter in the modularity function.

    Returns
    -------
    modularity: float
       The modularity on the normalized cocitation graph.
    """

    if type(adjacency) == sparse.csr_matrix:
        adj_matrix = adjacency
    elif sparse.isspmatrix(adjacency) or type(adjacency) == np.ndarray:
        adj_matrix = sparse.csr_matrix(adjacency)
    else:
        raise TypeError(
            "The argument must be a NumPy array or a SciPy Sparse matrix.")

    n_samples, n_features = adj_matrix.shape
    total_weight = adj_matrix.data.sum()
    pou = adj_matrix.dot(np.ones(n_samples)) / total_weight
    din = adj_matrix.T.dot(np.ones(n_features))

    # pseudo inverse square-root in-degree matrix
    dhin = sparse.diags(np.sqrt(din), shape=(n_features, n_features), format='csr')
    dhin.data = 1 / dhin.data

    normalized_adjacency = (adj_matrix.dot(dhin)).T.tocsr()

    if type(partition) == dict:
        labels = np.array([partition[i] for i in range(n_samples)])
    elif type(partition) == np.ndarray:
        labels = partition.copy()
    else:
        raise TypeError('The partition must be a dictionary or a NumPy array.')

    membership = sparse.csc_matrix((np.ones(n_samples), (np.arange(n_samples), labels)),
                                   shape=(n_samples, labels.max() + 1))
    fit = ((normalized_adjacency.dot(membership)).data ** 2).sum() / total_weight
    diversity = np.linalg.norm(membership.T.dot(pou)) ** 2

    return float(fit - resolution * diversity)


def performance(adjacency: sparse.csr_matrix, labels: np.ndarray) -> float:
    """The performance is the ratio of the total number of intra-cluster edges plus the total number of inter-cluster
    non-edges with the number of potential edges in the graph.

    Parameters
    ----------
    adjacency:
        the adjacency matrix of the graph
    labels:
        the cluster indices, labels[node] = index of the cluster of node.

    Returns
    -------
    : float
        the performance metric
    """
    n_nodes, m_nodes = adjacency.shape
    if n_nodes != m_nodes:
        raise ValueError('The adjacency matrix is not square.')
    bool_adj = abs(adjacency.sign())

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


def cocitation_performance(adjacency: sparse.csr_matrix, labels: np.ndarray) -> float:
    """The performance of the clustering on the normalized cocitation graph associated to the provided adjacency
    matrix without explicit computation of the graph.

    Parameters
    ----------
    adjacency:
        the adjacency matrix of the graph
    labels:
        the cluster indices, labels[node] = index of the cluster of node.

    Returns
    -------
    : float
        The performance metric on the normalized cocitation graph.
    """
    n_samples, n_features = adjacency.shape
    if n_samples != n_features:
        raise ValueError('The adjacency matrix is not square.')
    bool_adj = abs(adjacency.sign())

    clusters = set(labels.tolist())
    cluster_indicator = {cluster: (labels == cluster) for cluster in clusters}
    cluster_sizes = {cluster: cluster_indicator[cluster].sum() for cluster in clusters}

    din = bool_adj.T.dot(np.ones(n_features))
    # pseudo inverse square-root in-degree matrix
    dhin = sparse.diags(np.sqrt(din), shape=(n_features, n_features), format='csr')
    dhin.data = 1 / dhin.data

    norm_backward_adj = (dhin.dot(bool_adj.T)).tocsr()

    perf = 0.
    for cluster, indicator in cluster_indicator.items():
        perf += indicator.dot(bool_adj.dot(norm_backward_adj.dot(indicator)))

    for cluster_i, cluster_j in combinations(clusters, 2):
        perf += 2 * cluster_sizes[cluster_i] * cluster_sizes[cluster_j]
        perf -= 2 * cluster_indicator[cluster_i].dot(
            bool_adj.dot(norm_backward_adj.dot(cluster_indicator[cluster_j])))

    return perf / n_samples**2
