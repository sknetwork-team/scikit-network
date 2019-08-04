#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu July 10 2018
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""

import warnings
from typing import Union, Tuple

import numpy as np
from scipy import sparse

from sknetwork.utils.adjacency_formats import bipartite2undirected, directed2undirected
from sknetwork.utils.checks import check_format, check_probs


def modularity(adjacency: Union[sparse.csr_matrix, np.ndarray], labels: np.ndarray,
               secondary_labels: Union[None, np.ndarray] = None, weights: Union[str, np.ndarray] = 'degree',
               secondary_weights: Union[None, str, np.ndarray] = None, force_undirected: bool = False,
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
        Labels of each node, vector of size :math:`n` (or :math:`n_1`).
    secondary_labels:
        Label of each secondary node, vector of size :math:`n_2`, optional.
    weights :
        Weights (undirected graphs) or out-weights (directed graphs) of nodes.
        ``'degree'`` (default), ``'uniform'`` or custom weights.
    secondary_weights :
        Weights of secondary nodes (bipartite graphs) or in-weights of nodes (directed graphs).
        ``None`` (default), ``'degree'``, ``'uniform'`` or custom weights.
        If ``None``, taken equal to weights.
    force_undirected : bool (default= ``False``)
        If ``True``, consider the graph as undirected.
    force_biadjacency : bool (default= ``False``)
        If ``True``, force the input matrix to be considered as a biadjacency matrix.
        Only relevant for a symmetric input matrix.
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

    if secondary_labels is None:
        if n1 != n2:
            raise ValueError('For a biadjacency matrix, secondary labels must be specified.')
        else:
            secondary_labels = labels
    elif len(secondary_labels) != n2:
        raise ValueError('Dimension mismatch between secondary labels and biadjacency matrix.')

    _, unique_primary_labels = np.unique(labels, return_inverse=True)
    _, unique_secondary_labels = np.unique(secondary_labels, return_inverse=True)
    n_clusters = max(len(set(unique_primary_labels)), len(set(unique_secondary_labels)))

    primary_membership = sparse.csr_matrix((np.ones(n1), (np.arange(n1), unique_primary_labels)),
                                           shape=(n1, n_clusters))
    secondary_membership = sparse.csc_matrix((np.ones(n2), (np.arange(n2), unique_secondary_labels)),
                                             shape=(n2, n_clusters))

    fit: float = primary_membership.multiply(adjacency.dot(secondary_membership)).data.sum() / adjacency.data.sum()

    if n1 != n2 or force_biadjacency:
        # bipartite graph
        if force_undirected:
            adjacency = bipartite2undirected(adjacency)
            if type(weights) == str:
                probs = check_probs(weights, adjacency)
            else:
                probs = check_probs(np.append(weights, secondary_weights), adjacency)
            primary_div = primary_membership.T.dot(probs[:n1])
            secondary_div = secondary_membership.T.dot(probs[n1:])
            div: float = primary_div.dot(secondary_div)
            div = 2 * div + primary_div.dot(primary_div) + secondary_div.dot(secondary_div)
        else:
            primary_probs = check_probs(weights, adjacency)
            if secondary_weights is None:
                if type(weights) == str:
                    secondary_weights = weights
                else:
                    warnings.warn(Warning("Secondary_weights have been set to 'degree'."))
                    secondary_weights = 'degree'
            secondary_probs = check_probs(secondary_weights, adjacency.T)
            primary_div = primary_membership.T.dot(primary_probs)
            secondary_div = secondary_membership.T.dot(secondary_probs)
            div: float = primary_div.dot(secondary_div)
    else:
        # non-bipartite graph
        primary_probs = check_probs(weights, adjacency)
        secondary_probs = primary_probs
        primary_div = primary_membership.T.dot(primary_probs)
        secondary_div = secondary_membership.T.dot(secondary_probs)
        div: float = primary_div.dot(secondary_div)

    mod: float = fit - resolution * div
    if return_all:
        return mod, fit, div
    else:
        return mod


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
