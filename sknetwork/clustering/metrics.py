#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu July 10 2018
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""

from typing import Union, Tuple

import numpy as np
from scipy import sparse

from sknetwork.clustering.post_processing import membership_matrix
from sknetwork.linalg import diag_pinv
from sknetwork.utils.adjacency_formats import bipartite2directed
from sknetwork.utils.checks import check_format, check_probs, is_square


def modularity(adjacency: Union[sparse.csr_matrix, np.ndarray], labels: np.ndarray,
               weights: Union[str, np.ndarray] = 'degree', col_weights: Union[str, np.ndarray] = 'degree',
               resolution: float = 1, return_all: bool = False) -> Union[float, Tuple[float, float, float]]:
    """
    Compute the modularity of a clustering (node partition).

    The modularity of a clustering is

    :math:`Q = \\sum_{i,j}\\left(\\dfrac{A_{ij}}{w} - \\gamma \\dfrac{d_id_j}{w^2}\\right)\\delta_{c_i,c_j}`
    for undirected graphs,

    :math:`Q = \\sum_{i,j}\\left(\\dfrac{A_{ij}}{w} - \\gamma \\dfrac{d^+_id^-_j}{w^2}\\right)\\delta_{c_i,c_j}`
    for directed graphs,

    where

    :math:`A` is the adjacency matrix (size :math:`n\\times n)`,\n
    :math:`d_i` is the weight of node :math:`i` (undirected graphs),\n
    :math:`d^+_i, d^-_i` are the out-weight and in-weight of node :math:`i` (directed graphs),\n
    :math:`c_i` is the cluster of node :math:`i`,\n
    :math:`w = 1^TA1` is the total weight,\n
    :math:`\\delta` is the Kronecker symbol,\n
    :math:`\\gamma \\ge 0` is the resolution parameter.

    Parameters
    ----------
    adjacency:
        Adjacency or biadjacency matrix of the graph (shape :math:`n \\times n` or :math:`n_1 \\times n_2`).
    labels:
        Labels of nodes, vector of size :math:`n` (or :math:`n_1` for bipartite graphs).
    weights :
        Weights of nodes.
        ``'degree'`` (default), ``'uniform'`` or custom weights.
    col_weights :
        Weights of secondary nodes (for bipartite graphs).
        ``None`` (default), ``'degree'``, ``'uniform'`` or custom weights.
        If ``None``, taken equal to weights.
    resolution:
        Resolution parameter (default = 1).
    return_all:
        If ``True``, return modularity, fit, diversity.

    Returns
    -------
    modularity : float
    fit: float, optional
    diversity: float, optional

    """

    adjacency = check_format(adjacency).astype(float)
    if not is_square(adjacency):
        raise ValueError('The adjacency is not square.')

    if len(labels) != adjacency.shape[0]:
        raise ValueError('Dimension mismatch between labels and adjacency matrix.')

    row_probs = check_probs(weights, adjacency)
    col_probs = check_probs(col_weights, adjacency.T)
    membership = membership_matrix(labels)

    fit: float = membership.multiply(adjacency.dot(membership)).data.sum() / adjacency.data.sum()
    div: float = membership.T.dot(col_probs).dot(membership.T.dot(row_probs))
    mod: float = fit - resolution * div
    if return_all:
        return mod, fit, div
    else:
        return mod


def bimodularity(biadjacency: Union[sparse.csr_matrix, np.ndarray], labels: np.ndarray, col_labels: np.ndarray,
                 weights: Union[str, np.ndarray] = 'degree', col_weights: Union[str, np.ndarray] = 'degree',
                 resolution: float = 1, return_all: bool = False) -> Union[float, Tuple[float, float, float]]:
    """
    Compute the bimodularity of a clustering (node partition).

    The bimodularity of a clustering is

    :math:`Q = \\sum_{i}\\sum_{j}\\left(\\dfrac{B_{ij}}{w} - \\gamma \\dfrac{d_if_j}{w^2}\\right)
    \\delta_{c_i,c_j}`

    where

    :math:`B` is the biadjacency matrix (size :math:`n_1\\times n_2)`,\n
    :math:`d_i, f_j` are the weights of sample node :math:`i` (row) and feature node :math:`j` (column),\n
    :math:`c_i, c_j` are the clusters of sample node :math:`i` (row) and feature node :math:`j` (column),\n
    :math:`w = 1^TB1` is the total weight,\n
    :math:`\\delta` is the Kronecker symbol,\n
    :math:`\\gamma \\ge 0` is the resolution parameter.

    Parameters
    ----------
    biadjacency:
        Biadjacency matrix of the graph (shape :math:`n_1 \\times n_2`).
    labels:
        Labels of row nodes, vector of size :math:`n1`.
    col_labels:
        Labels of column nodes, vector of size :math:`n_2`.
    weights :
        Weights of nodes.
        ``'degree'`` (default), ``'uniform'`` or custom weights.
    col_weights :
        Weights of column nodes.
        ``'degree'`` (default), ``'uniform'`` or custom weights.
    resolution:
        Resolution parameter (default = 1).
    return_all:
        If ``True``, return modularity, fit, diversity.

    Returns
    -------
    modularity : float
    fit: float, optional
    diversity: float, optional
    """

    biadjacency = check_format(biadjacency).astype(float)
    n1, n2 = biadjacency.shape

    if len(labels) != n1:
        raise ValueError('Dimension mismatch between labels and biadjacency matrix.')
    if len(col_labels) != n2:
        raise ValueError('Dimension mismatch between col_labels and biadjacency matrix.')

    adjacency = bipartite2directed(biadjacency)

    new_weights = check_probs(weights, biadjacency)
    new_weights = np.hstack((new_weights, np.zeros(n2)))
    new_col_weights = check_probs(col_weights, biadjacency.T)
    new_col_weights = np.hstack((np.zeros(n1), new_col_weights))

    new_labels = np.hstack((labels, col_labels))

    return modularity(adjacency, new_labels, new_weights, new_col_weights, resolution, return_all)


def cocitation_modularity(adjacency: Union[sparse.csr_matrix, np.ndarray], labels: np.ndarray, resolution: float = 1,
                          return_all: bool = False) -> Union[float, Tuple[float, float, float]]:
    """
    Computes the modularity of a clustering in the normalized cocitation graph.
    Does not require the explicit computation of the normalized cocitation adjacency matrix.

    :math:`Q = \\sum_{i,j}\\left(\\dfrac{(AF^{-1}A^T)_{ij}}{w} - \\gamma \\dfrac{d_id_j}{w^2}\\right)
    \\delta_{c_i,c_j}`

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
    return_all:
        If ``True``, return modularity, fit, diversity.

    Returns
    -------
    modularity : float
    fit: float, optional
    diversity: float, optional
    """

    adjacency = check_format(adjacency).astype(float)

    n1, n2 = adjacency.shape
    total_weight = adjacency.data.sum()
    probs = adjacency.dot(np.ones(n2)) / total_weight

    col_weights = adjacency.T.dot(np.ones(n2))
    col_diag = diag_pinv(np.sqrt(col_weights))
    normalized_adjacency = (adjacency.dot(col_diag)).T.tocsr()

    if len(labels) != n1:
        raise ValueError('The number of labels must match the number of rows.')

    membership = membership_matrix(labels)
    fit: float = ((normalized_adjacency.dot(membership)).data ** 2).sum() / total_weight
    div: float = np.linalg.norm(membership.T.dot(probs)) ** 2
    mod: float = fit - resolution * div

    if return_all:
        return mod, fit, div
    else:
        return mod


def nsd(labels: np.ndarray) -> float:
    """Normalized Standard Deviation in cluster size.

    The metric is normalized so that a perfectly balanced clustering yields a score of 1 while unbalanced ones yield
    scores close to 0.

    Parameters
    ----------
    labels:
        Labels of nodes.

    Returns
    -------
    nsd: float

    """

    n = labels.shape[0]
    _, counts = np.unique(labels, return_counts=True)
    k = counts.shape[0]
    if k == 0:
        raise ValueError('There must be at least two different clusters.')
    return 1 - np.std(counts) / np.sqrt(n ** 2 * (k - 1) / k ** 2)
