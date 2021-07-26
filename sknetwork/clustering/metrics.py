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

from sknetwork.linalg.normalization import diag_pinv
from sknetwork.utils.check import check_format, get_probs, check_square
from sknetwork.utils.format import bipartite2directed
from sknetwork.utils.membership import membership_matrix


def modularity(adjacency: Union[sparse.csr_matrix, np.ndarray], labels: np.ndarray,
               weights: Union[str, np.ndarray] = 'degree', weights_in: Union[str, np.ndarray] = 'degree',
               resolution: float = 1, return_all: bool = False) -> Union[float, Tuple[float, float, float]]:
    """Modularity of a clustering.

    The modularity of a clustering is

    :math:`Q = \\dfrac{1}{w} \\sum_{i,j}\\left(A_{ij} - \\gamma \\dfrac{d_id_j}{w}\\right)\\delta_{c_i,c_j}`
    for graphs,

    :math:`Q = \\dfrac{1}{w} \\sum_{i,j}\\left(A_{ij} - \\gamma \\dfrac{d^+_id^-_j}{w}\\right)\\delta_{c_i,c_j}`
    for directed graphs,

    where

    * :math:`c_i` is the cluster of node :math:`i`,\n
    * :math:`d_i` is the weight of node :math:`i`,\n
    * :math:`d^+_i, d^-_i` are the out-weight, in-weight of node :math:`i` (for digraphs),\n
    * :math:`w = 1^TA1` is the total weight,\n
    * :math:`\\delta` is the Kronecker symbol,\n
    * :math:`\\gamma \\ge 0` is the resolution parameter.

    Parameters
    ----------
    adjacency:
        Adjacency matrix of the graph.
    labels:
        Labels of nodes, vector of size :math:`n` .
    weights :
        Weights of nodes.
        ``'degree'`` (default), ``'uniform'`` or custom weights.
    weights_in :
        In-weights of nodes.
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

    Example
    -------
    >>> from sknetwork.clustering import modularity
    >>> from sknetwork.data import house
    >>> adjacency = house()
    >>> labels = np.array([0, 0, 1, 1, 0])
    >>> np.round(modularity(adjacency, labels), 2)
    0.11
    """
    adjacency = check_format(adjacency).astype(float)
    check_square(adjacency)

    if len(labels) != adjacency.shape[0]:
        raise ValueError('Dimension mismatch between labels and adjacency matrix.')

    probs_row = get_probs(weights, adjacency)
    probs_col = get_probs(weights_in, adjacency.T)
    membership = membership_matrix(labels)

    fit: float = membership.multiply(adjacency.dot(membership)).data.sum() / adjacency.data.sum()
    div: float = membership.T.dot(probs_col).dot(membership.T.dot(probs_row))
    mod: float = fit - resolution * div
    if return_all:
        return mod, fit, div
    else:
        return mod


def bimodularity(biadjacency: Union[sparse.csr_matrix, np.ndarray], labels: np.ndarray, labels_col: np.ndarray,
                 weights: Union[str, np.ndarray] = 'degree', weights_col: Union[str, np.ndarray] = 'degree',
                 resolution: float = 1, return_all: bool = False) -> Union[float, Tuple[float, float, float]]:
    """Bimodularity of the clustering (for bipartite graphs).

    The bimodularity of a clustering is

    :math:`Q = \\sum_{i}\\sum_{j}\\left(\\dfrac{B_{ij}}{w} - \\gamma \\dfrac{d_{1,i}d_{2,j}}{w^2}\\right)
    \\delta_{c_{1,i},c_{2,j}}`

    where

    * :math:`c_{1,i}, c_{2,j}` are the clusters of nodes :math:`i` (row) and :math:`j` (column),\n
    * :math:`d_{1,i}, d_{2,j}` are the weights of nodes :math:`i` (row) and :math:`j` (column),\n
    * :math:`w = 1^TB1` is the total weight,\n
    * :math:`\\delta` is the Kronecker symbol,\n
    * :math:`\\gamma \\ge 0` is the resolution parameter.

    Parameters
    ----------
    biadjacency :
        Biadjacency matrix of the graph (shape :math:`n_1 \\times n_2`).
    labels :
        Labels of rows, vector of size :math:`n_1`.
    labels_col:
        Labels of columns, vector of size :math:`n_2`.
    weights :
        Weights of nodes.
        ``'degree'`` (default), ``'uniform'`` or custom weights.
    weights_col :
        Weights of columns.
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

    Example
    -------
    >>> from sknetwork.clustering import bimodularity
    >>> from sknetwork.data import star_wars
    >>> biadjacency = star_wars()
    >>> labels = np.array([1, 1, 0, 0])
    >>> labels_col = np.array([1, 0, 0])
    >>> np.round(bimodularity(biadjacency, labels, labels_col), 2)
    0.22
    """
    biadjacency = check_format(biadjacency).astype(float)
    n_row, n_col = biadjacency.shape

    if len(labels) != n_row:
        raise ValueError('Dimension mismatch between labels and biadjacency matrix.')
    if len(labels_col) != n_col:
        raise ValueError('Dimension mismatch between labels_col and biadjacency matrix.')

    adjacency = bipartite2directed(biadjacency)

    weights_ = get_probs(weights, biadjacency)
    weights_ = np.hstack((weights_, np.zeros(n_col)))
    weights_col_ = get_probs(weights_col, biadjacency.T)
    weights_col_ = np.hstack((np.zeros(n_row), weights_col_))

    labels_ = np.hstack((labels, labels_col))

    return modularity(adjacency, labels_, weights_, weights_col_, resolution, return_all)


def comodularity(adjacency: Union[sparse.csr_matrix, np.ndarray], labels: np.ndarray, resolution: float = 1,
                 return_all: bool = False) -> Union[float, Tuple[float, float, float]]:
    """Modularity of a clustering in the normalized co-neighborhood graph.

    Quality metric of a clustering given by:

    :math:`Q = \\dfrac{1}{w}\\sum_{i,j}\\left((AD_2^{-1}A^T)_{ij} - \\gamma \\dfrac{d_id_j}{w}\\right)
    \\delta_{c_i,c_j}`

    where

    * :math:`c_i` is the cluster of node `i`,\n
    * :math:`D_2` is the diagonal matrix of the weights of columns,\n
    * :math:`\\delta` is the Kronecker symbol,\n
    * :math:`\\gamma \\ge 0` is the resolution parameter.

    Parameters
    ----------
    adjacency :
        Adjacency matrix or biadjacency matrix of the graph.
    labels :
       Labels of the nodes.
    resolution :
        Resolution parameter (default = 1).
    return_all :
        If ``True``, return modularity, fit, diversity.

    Returns
    -------
    modularity : float
    fit : float, optional
    diversity: float, optional

    Example
    -------
    >>> from sknetwork.clustering import comodularity
    >>> from sknetwork.data import house
    >>> adjacency = house()
    >>> labels = np.array([0, 0, 1, 1, 0])
    >>> np.round(comodularity(adjacency, labels), 2)
    0.06

    Notes
    -----
    Does not require the computation of the adjacency matrix of the normalized co-neighborhood graph.
    """
    adjacency = check_format(adjacency).astype(float)

    n_row, n_col = adjacency.shape
    total_weight = adjacency.data.sum()
    probs = adjacency.dot(np.ones(n_col)) / total_weight

    weights_col = adjacency.T.dot(np.ones(n_col))
    diag_col = diag_pinv(np.sqrt(weights_col))
    normalized_adjacency = (adjacency.dot(diag_col)).T.tocsr()

    if len(labels) != n_row:
        raise ValueError('The number of labels must match the number of rows.')

    membership = membership_matrix(labels)
    fit: float = ((normalized_adjacency.dot(membership)).data ** 2).sum() / total_weight
    div: float = np.linalg.norm(membership.T.dot(probs)) ** 2
    mod: float = fit - resolution * div

    if return_all:
        return mod, fit, div
    else:
        return mod


def normalized_std(labels: np.ndarray) -> float:
    """Normalized standard deviation of cluster sizes.

    A score of 1 means perfectly balanced clustering.

    Parameters
    ----------
    labels :
        Labels of nodes.

    Returns
    -------
    float

    Example
    -------
    >>> from sknetwork.clustering import normalized_std
    >>> labels = np.array([0, 0, 1, 1])
    >>> normalized_std(labels)
    1.0
    """
    n = labels.shape[0]
    unique_clusters, counts = np.unique(labels, return_counts=True)
    n_clust = len(unique_clusters)
    if n_clust < 2:
        raise ValueError('There must be at least two different clusters.')
    return 1 - np.std(counts) / np.sqrt(n ** 2 * (n_clust - 1) / n_clust ** 2)
