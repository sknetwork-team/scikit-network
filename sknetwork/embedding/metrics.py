#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 6 2018
@author: Nathan De Lara <ndelara@enst.fr>
Quality metrics for adjacency embeddings
"""

import numpy as np
from scipy import sparse

from sknetwork.linalg import normalize
from sknetwork.utils.check import check_format, check_probs, is_square


def cosine_modularity(adjacency, embedding: np.ndarray, embedding_col=None, resolution=1., weights='degree',
                      return_all: bool = False):
    """Quality metric of an embedding :math:`x` defined by:

    :math:`Q = \\sum_{ij}\\left(\\dfrac{A_{ij}}{w} - \\gamma \\dfrac{w_iw'_j}{w^2}\\right)
    \\left(\\dfrac{1 + \\pi(x_i)^T\\pi(x_j)}{2}\\right)`

    where :math:`\\pi(x_i)` is the projection of :math:`x_i` onto the unit-sphere.

    For bipartite graphs with column embedding :math:`y`, the metric is

    :math:`Q = \\sum_{ij}\\left(\\dfrac{B_{ij}}{w} - \\gamma \\dfrac{w_iw'_j}{w^2}\\right)
    \\left(\\dfrac{1 + \\pi(x_i)^T\\pi(y_j)}{2}\\right)`

    This metric is normalized to lie between -1 and 1 (for :math:`\\gamma = 1`).

    Parameters
    ----------
    adjacency: sparse.csr_matrix or np.ndarray
        Adjacency matrix of the graph.
    embedding: np.ndarray
        Embedding of the nodes.
    embedding_col: None or np.ndarray
        For biadjacency matrices, embedding of the columns.
    resolution: float
        Resolution parameter.
    weights: ``'degree'`` or ``'uniform'``
        Weights of the nodes.
    return_all: bool, default = ``False``
        whether to return (fit, div, :math:`Q`) or :math:`Q`

    Returns
    -------
    modularity : float
    fit: float, optional
    diversity: float, optional
    """
    adjacency = check_format(adjacency)
    total_weight: float = adjacency.data.sum()

    if embedding_col is None:
        if not is_square(adjacency):
            raise ValueError('embedding_col cannot be None for non-square adjacency matrices.')
        else:
            embedding_col = embedding.copy()

    embedding_row_norm = normalize(embedding, p=1)
    embedding_col_norm = normalize(embedding_col, p=1)

    probs_row = check_probs(weights, adjacency)
    probs_col = check_probs(weights, adjacency.T)

    fit: float = 0.5 * (1 + (np.multiply(embedding_row_norm, adjacency.dot(embedding_col_norm))).sum() / total_weight)
    div: float = 0.5 * (1 + (embedding.T.dot(probs_row)).dot(embedding_col.T.dot(probs_col)))

    if return_all:
        return fit, div, fit - resolution * div
    else:
        return fit - resolution * div
