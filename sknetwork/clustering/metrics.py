#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu July 10 2018
@author: Nathan de Lara <ndelara@enst.fr>
"""

import numpy as np
from scipy import sparse
from typing import Union


def modularity(adjacency: Union[sparse.csr_matrix, np.ndarray], partition: Union[dict, np.ndarray],
               resolution: float = 1) -> float:
    """
    Compute the modularity of a clustering (node partition).

    The modularity of a clustering is

    :math:`Q = \\sum_{i,j=1}^n\\big(\\dfrac{A_{ij}}{w} - \\gamma \\dfrac{d_id_j}{w^2}\\big)\\delta_{c_i,c_j}`
    for undirected graphs

    :math:`Q = \\sum_{i,j=1}^n\\big(\\dfrac{A_{ij}}{w} - \\gamma \\dfrac{d^+_id^-_j}{w^2}\\big)\\delta_{c_i,c_j}`
    for directed graphs

    where

    :math:`c_i` is the cluster of node `i`,\n
    :math:`\\delta` is the Kronecker symbol,\n
    :math:`\\gamma \\ge 0` is the resolution parameter.

    Parameters
    ----------
    partition : dict or np.ndarray
       The partition of the nodes. The keys of the dictionary correspond to the nodes and the values to the labels.
    adjacency : scipy.csr_matrix or np.ndarray
        The adjacency matrix of the graph (sparse or dense).
    resolution : float, optional (default=1.)
        The resolution parameter.

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

    n, p = adj_matrix.shape
    if n != p:
        raise ValueError('The adjacency must be a square matrix.')
    norm_adj = adj_matrix / adj_matrix.data.sum()
    out_probs = norm_adj.dot(np.ones(n))
    in_probs = norm_adj.T.dot(np.ones(n))

    if type(partition) == dict:
        labels = np.array([partition[i] for i in range(n)])
    elif type(partition) == np.ndarray:
        labels = partition.copy()
    else:
        raise TypeError('The partition must be a dictionary or a NumPy array.')

    row = np.arange(n)
    col = labels
    data = np.ones(n)
    membership = sparse.csr_matrix((data, (row, col)))

    fit = ((membership.multiply(norm_adj.dot(membership))).dot(np.ones(membership.shape[1]))).sum()
    diversity = membership.T.dot(in_probs).dot(membership.T.dot(out_probs))
    return float(fit - resolution * diversity)


def bimodularity(biadjacency: sparse.csr_matrix, sample_labels: np.ndarray, feature_labels: np.ndarray,
                 resolution: float = 1) -> float:
    """
    Modularity for bipartite graphs:

    :math:`Q = \\sum_{i=1}^n\\sum_{j=1}^p (\\dfrac{B_{ij}}{w} - \\gamma \\dfrac{d_if_j}{w^2})\\delta_{c^d_i,c^f_j},`

    where

    :math:`B` is the biadjacency matrix of the graph (shape :math:`n\\times p`),\n
    :math:`c^d_i` is the cluster of sample node `i` (rows of the biadjacency matrix),\n
    :math:`c^f_j` is the cluster of feature node `j` (columns of the biadjacency matrix),\n
    :math:`\\delta` is the Kronecker symbol,\n
    :math:`\\gamma \\ge 0` is the resolution parameter.


    Parameters
    ----------
    biadjacency:
        Matrix of shape n x p.
    sample_labels:
        Cluster of each sample, vector of size n.
    feature_labels:
        Cluster of each feature, vector of size p.
    resolution:
        Resolution parameter.

    Returns
    -------
    bimodularity:
        Bimodularity of the clustering.
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


def cocitation_modularity(adjacency, partition, resolution: float = 1) -> float:
    """
    Computes the modularity of a clustering on the normalized cocitation adjacency matrix.
    Does not require the explicit computation of the normalized cocitation adjacency matrix.

    :math:`Q = \\sum_{i,j=1}^n(\\dfrac{(AF^{-1}A^T)_{ij}}{w} - \\gamma \\dfrac{d_id_j}{w^2})\\delta_{c_i,c_j}`

    where

    :math:`AF^{-1}A^T` is the normalized cocitation adjacency matrix,\n
    :math:`c_j` is the cluster of node `i`,\n
    :math:`\\delta` is the Kronecker symbol,\n
    :math:`\\gamma \\ge 0` is the resolution parameter.

    Parameters
    ----------
    partition: dict or np.ndarray
       The partition of the nodes. The keys of the dictionary correspond to the nodes and the values to the communities.
    adjacency: scipy.csr_matrix or np.ndarray
        The adjacency matrix of the adjacency (sparse or dense).
    resolution: float (default=1.)
        The resolution parameter.

    Returns
    -------
    modularity: float
       The modularity on the normalized cocitation adjacency matrix.
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

