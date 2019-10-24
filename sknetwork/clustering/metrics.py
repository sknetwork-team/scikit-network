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
from sknetwork.utils.adjacency_formats import set_adjacency_weights
from sknetwork.utils.checks import check_format


def modularity(adjacency: Union[sparse.csr_matrix, np.ndarray], labels: np.ndarray,
               col_labels: Union[None, np.ndarray] = None, weights: Union[str, np.ndarray] = 'degree',
               col_weights: Union[None, str, np.ndarray] = None, force_undirected: bool = False,
               force_biadjacency: bool = False, resolution: float = 1, return_all: bool = False) \
                    -> Union[float, Tuple[float, float, float]]:
    """
    Compute the modularity of a clustering (node partition).

    The modularity of a clustering is

    :math:`Q = \\sum_{i,j=1}^n\\big(\\dfrac{A_{ij}}{w} - \\gamma \\dfrac{w_iw_j}{w^2}\\big)\\delta_{c_i,c_j}`
    for undirected graphs,

    :math:`Q = \\sum_{i,j=1}^n\\big(\\dfrac{A_{ij}}{w} - \\gamma \\dfrac{w^+_iw^-_j}{w^2}\\big)\\delta_{c_i,c_j}`
    for directed graphs,

    :math:`Q = \\sum_{i=1}^{n_1}\\sum_{j=1}^{n_2}\\big(\\dfrac{B_{ij}}{w} - \\gamma \\dfrac{w^1_iw^2_j}{w^2}\\big)
    \\delta_{c^1_i,c^2_j}`
    for bipartite graphs,

    where

    :math:`A` is the adjacency matrix (size :math:`n\\times n)`,\n
    :math:`w_i` is the weight of node :math:`i` (undirected graphs),\n
    :math:`w^+_i, w^-_i` are the out-weight and in-weight of node :math:`i` (directed graphs),\n
    :math:`c_i` is the cluster of node :math:`i` (undirected and directed graphs),\n
    :math:`B` is the biadjacency matrix (size :math:`n_1\\times n_2)`,\n
    :math:`w^1_i, w^2_j` are the weights of primary node :math:`i` and secondary node :math:`j` (bipartite graphs),\n
    :math:`c^1_i, c^2_j` are the clusters of primary node :math:`i` and secondary node :math:`j` (bipartite graphs),\n
    :math:`w = 1^TA1` or :math:`w = 1^TB1` is the total weight,\n
    :math:`\\delta` is the Kronecker symbol,\n
    :math:`\\gamma \\ge 0` is the resolution parameter.

    Parameters
    ----------
    adjacency:
        Adjacency or biadjacency matrix of the graph (shape :math:`n \\times n` or :math:`n_1 \\times n_2`).
    labels:
        Labels of nodes, vector of size :math:`n` (or :math:`n_1` for bipartite graphs
        ).
    col_labels:
        Labels of secondary nodes (for bipartite graphs), vector of size :math:`n_2`.
    weights :
        Weights of nodes.
        ``'degree'`` (default), ``'uniform'`` or custom weights.
    col_weights :
        Weights of secondary nodes (for bipartite graphs).
        ``None`` (default), ``'degree'``, ``'uniform'`` or custom weights.
        If ``None``, taken equal to weights.
    force_undirected : bool (default= ``False``)
        If ``True``, consider the graph as undirected.
    force_biadjacency : bool (default= ``False``)
        If ``True``, force the input matrix to be considered as a biadjacency matrix.
    resolution:
        Resolution parameter (default = 1).
    return_all:
        If ``True``, return modularity, fit, diversity.

    Returns
    -------
    modularity : float or Tuple
        Modularity (with fit and diversity if **return_all** = ``True``)
    """

    adjacency = check_format(adjacency)
    n1, n2 = adjacency.shape

    if len(labels) != n1:
        raise ValueError('Dimension mismatch between labels and adjacency matrix.')

    if n1 != n2 or force_biadjacency:
        if col_labels is None:
            raise ValueError('For a biadjacency matrix, col_labels must be specified.')
        elif len(labels) != n1:
            raise ValueError('Dimension mismatch between col_labels and biadjacency matrix.')
        else:
            labels = np.hstack((labels, col_labels))

    adjacency, row_weights, col_weights = set_adjacency_weights(adjacency, weights, col_weights, force_undirected,
                                                                force_biadjacency)
    membership = membership_matrix(labels)

    fit: float = membership.multiply(adjacency.dot(membership)).data.sum() / adjacency.data.sum()
    div: float = membership.T.dot(col_weights).dot(membership.T.dot(row_weights))
    mod: float = fit - resolution * div
    if return_all:
        return mod, fit, div
    else:
        return mod


def cocitation_modularity(adjacency: Union[sparse.csr_matrix, np.ndarray], labels: np.ndarray,
                          resolution: float = 1, return_all: bool = False) \
                    -> Union[float, Tuple[float, float, float]]:
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
    return_all:
        If ``True``, return modularity, fit, diversity.

    Returns
    -------
    modularity: float or Tuple
       Modularity of the clustering in the normalized cocitation graph
       (with fit and diversity if **return_all** = ``True``).
    """

    adjacency = check_format(adjacency)

    n1, n2 = adjacency.shape
    total_weight = adjacency.data.sum()
    probs = adjacency.dot(np.ones(n2)) / total_weight

    col_weights = adjacency.T.dot(np.ones(n2))

    # pseudo inverse square-root feature weight matrix
    norm_diag_matrix = sparse.diags(np.sqrt(col_weights), shape=(n2, n2), format='csr')
    norm_diag_matrix.data = 1 / norm_diag_matrix.data

    normalized_adjacency = (adjacency.dot(norm_diag_matrix)).T.tocsr()

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
