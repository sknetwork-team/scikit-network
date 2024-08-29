#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in July 2018
@author: Nathan de Lara <nathan.delara@polytechnique.org>
@author: Thomas Bonald <bonald@enst.fr>
"""
from typing import Optional, Union, Tuple

import numpy as np
from scipy import sparse

from sknetwork.utils.check import get_probs
from sknetwork.utils.format import get_adjacency
from sknetwork.utils.membership import get_membership


def get_modularity(input_matrix: Union[sparse.csr_matrix, np.ndarray], labels: np.ndarray,
                   labels_col: Optional[np.ndarray] = None, weights: str = 'degree',
                   resolution: float = 1, return_all: bool = False) -> Union[float, Tuple[float, float, float]]:
    """Modularity of a clustering.

    The modularity of a clustering is

    :math:`Q = \\dfrac{1}{w} \\sum_{i,j}\\left(A_{ij} - \\gamma \\dfrac{w_iw_j}{w}\\right)\\delta_{c_i,c_j}`
    for graphs,

    :math:`Q = \\dfrac{1}{w} \\sum_{i,j}\\left(A_{ij} - \\gamma \\dfrac{d^+_id^-_j}{w}\\right)\\delta_{c_i,c_j}`
    for directed graphs,

    where

    * :math:`c_i` is the cluster of node :math:`i`,\n
    * :math:`w_i` is the weight of node :math:`i`,\n
    * :math:`w^+_i, w^-_i` are the out-weight, in-weight of node :math:`i` (for directed graphs),\n
    * :math:`w = 1^TA1` is the total weight,\n
    * :math:`\\delta` is the Kronecker symbol,\n
    * :math:`\\gamma \\ge 0` is the resolution parameter.

    Parameters
    ----------
    input_matrix :
        Adjacency matrix or biadjacency matrix of the graph.
    labels :
        Labels of nodes.
    labels_col :
        Labels of column nodes (for bipartite graphs).
    weights :
        Weighting of nodes (``'degree'`` (default) or ``'uniform'``).
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
    >>> from sknetwork.clustering import get_modularity
    >>> from sknetwork.data import house
    >>> adjacency = house()
    >>> labels = np.array([0, 0, 1, 1, 0])
    >>> float(np.round(get_modularity(adjacency, labels), 2))
    0.11
    """
    adjacency, bipartite = get_adjacency(input_matrix.astype(float))

    if bipartite:
        if labels_col is None:
            raise ValueError('For bipartite graphs, you must specify the labels of both rows and columns.')
        else:
            labels = np.hstack((labels, labels_col))

    if len(labels) != adjacency.shape[0]:
        raise ValueError('Dimension mismatch between labels and input matrix.')

    probs_row = get_probs(weights, adjacency)
    probs_col = get_probs(weights, adjacency.T)
    membership = get_membership(labels).astype(float)

    fit = membership.T.dot(adjacency.dot(membership)).diagonal().sum() / adjacency.data.sum()
    div = membership.T.dot(probs_col).dot(membership.T.dot(probs_row))
    mod = fit - resolution * div
    if return_all:
        return mod, fit, div
    else:
        return mod
