#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu July 10 2018
@author: Nathan de Lara <ndelara@enst.fr>
"""

import numpy as np
from scipy import sparse
from typing import Union
from sknetwork.utils.checks import check_format, is_square


def modularity(adjacency: Union[sparse.csr_matrix, np.ndarray], labels: np.ndarray,
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
    adjacency:
        Adjacency matrix of the graph.
    labels:
        Labels of the nodes.
    resolution:
        Resolution parameter (default = 1).

    Returns
    -------
    modularity : float
        The modularity.
    """

    adjacency = check_format(adjacency)

    if not is_square(adjacency):
        raise ValueError('The adjacency must be a square matrix.')

    n = adjacency.shape[0]
    norm_adj = adjacency / adjacency.data.sum()
    out_probs = norm_adj.dot(np.ones(n))
    in_probs = norm_adj.T.dot(np.ones(n))

    if len(labels) != n:
        raise ValueError('The number of labels must match the number of rows.')

    row = np.arange(n)
    col = labels
    data = np.ones(n)
    membership = sparse.csr_matrix((data, (row, col)))

    fit = ((membership.multiply(norm_adj.dot(membership))).dot(np.ones(membership.shape[1]))).sum()
    diversity = membership.T.dot(in_probs).dot(membership.T.dot(out_probs))
    return float(fit - resolution * diversity)


def bimodularity(biadjacency: Union[sparse.csr_matrix, np.ndarray], sample_labels: np.ndarray,
                 feature_labels: np.ndarray, resolution: float = 1) -> float:
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
        Biadjacency matrix of the graph (shape n x p).
    sample_labels:
        Label of each sample, vector of size n.
    feature_labels:
        Cluster of each feature, vector of size p.
    resolution:
        Resolution parameter.

    Returns
    -------
    bimodularity: float
        Bimodularity of the clustering.
    """
    biadjacency = check_format(biadjacency)

    n, p = biadjacency.shape
    total_weight: float = biadjacency.data.sum()
    sample_weights = biadjacency.dot(np.ones(p)) / total_weight
    features_weights = biadjacency.T.dot(np.ones(n)) / total_weight

    if len(sample_labels) != n or len(feature_labels) != p:
        raise ValueError('The number of sample / feature labels must match the number of rows / columns.')

    _, unique_sample_labels = np.unique(sample_labels, return_inverse=True)
    _, unique_feature_labels = np.unique(feature_labels, return_inverse=True)
    n_clusters = max(len(set(unique_sample_labels)), len(set(unique_feature_labels)))

    sample_membership = sparse.csr_matrix((np.ones(n), (np.arange(n), unique_sample_labels)),
                                          shape=(n, n_clusters))
    feature_membership = sparse.csc_matrix((np.ones(p), (np.arange(p), unique_feature_labels)),
                                           shape=(p, n_clusters))

    fit: float = sample_membership.multiply(biadjacency.dot(feature_membership)).data.sum() / total_weight
    div: float = (sample_membership.T.dot(sample_weights)).dot(feature_membership.T.dot(features_weights))

    return fit - resolution * div


def cocitation_modularity(adjacency: Union[sparse.csr_matrix, np.ndarray], labels: np.ndarray,
                          resolution: float = 1) -> float:
    """
    Computes the modularity of a clustering in the normalized cocitation graph.
    Does not require the explicit computation of the normalized cocitation adjacency matrix.

    :math:`Q = \\sum_{i,j=1}^n(\\dfrac{(AF^{-1}A^T)_{ij}}{w} - \\gamma \\dfrac{d_id_j}{w^2})\\delta_{c_i,c_j}`

    where

    :math:`AF^{-1}A^T` is the normalized cocitation adjacency matrix,\n
    :math:`F = \\text{diag}(A^T1)` is the diagonal matrix of feature weights,\n
    :math:`c_i` is the cluster of node `i`,\n
    :math:`\\delta` is the Kronecker symbol,\n
    :math:`\\gamma \\ge 0` is the resolution parameter.

    Parameters
    ----------
    adjacency:
        Adjacency matrix of the graph.
    labels:
       Labels of the nodes.
    resolution:
        Resolution parameter (default = 1).

    Returns
    -------
    modularity: float
       Modularity of the clustering in the normalized cocitation graph.
    """

    adjacency = check_format(adjacency)

    n, p = adjacency.shape
    total_weight = adjacency.data.sum()
    probs = adjacency.dot(np.ones(n)) / total_weight

    feature_weights = adjacency.T.dot(np.ones(p))

    # pseudo inverse square-root feature weight matrix
    norm_diag_matrix = sparse.diags(np.sqrt(feature_weights), shape=(p, p), format='csr')
    norm_diag_matrix.data = 1 / norm_diag_matrix.data

    normalized_adjacency = (adjacency.dot(norm_diag_matrix)).T.tocsr()

    if len(labels) != n:
        raise ValueError('The number of labels must match the number of rows.')

    membership = sparse.csc_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, labels.max() + 1))
    fit = ((normalized_adjacency.dot(membership)).data ** 2).sum() / total_weight
    diversity = np.linalg.norm(membership.T.dot(probs)) ** 2

    return float(fit - resolution * diversity)
