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
from sknetwork.utils.check import check_format, check_probs, check_square


def cosine_modularity(adjacency, embedding: np.ndarray, embedding_col=None, resolution=1., weights='degree',
                      return_all: bool = False):
    """Quality metric of an embedding :math:`x` defined by:

    :math:`Q = \\sum_{ij}\\left(\\dfrac{A_{ij}}{w} - \\gamma \\dfrac{w^+_iw^-_j}{w^2}\\right)
    \\left(\\dfrac{1 + \\cos(x_i, x_j)}{2}\\right)`

    where

    * :math:`w^+_i, w^-_i` are the out-weight, in-weight of node :math:`i` (for digraphs),\n
    * :math:`w = 1^TA1` is the total weight of the graph.

    For bipartite graphs with column embedding :math:`y`, the metric is

    :math:`Q = \\sum_{ij}\\left(\\dfrac{B_{ij}}{w} - \\gamma \\dfrac{w_{1,i}w_{2,j}}{w^2}\\right)
    \\left(\\dfrac{1 + \\cos(x_i, y_j)}{2}\\right)`

    where

    * :math:`w_{1,i}, w_{2,j}` are the weights of nodes :math:`i` (row) and :math:`j` (column),\n
    * :math:`w = 1^TB1` is the total weight of the graph.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    embedding :
        Embedding of the nodes.
    embedding_col :
        Embedding of the columns (for bipartite graphs).
    resolution :
        Resolution parameter.
    weights : ``'degree'`` or ``'uniform'``
        Weights of the nodes.
    return_all :
        If ``True``, also return fit and diversity

    Returns
    -------
    modularity : float
    fit: float, optional
    diversity: float, optional

    Example
    -------
    >>> from sknetwork.embedding import cosine_modularity
    >>> from sknetwork.data import karate_club
    >>> graph = karate_club(metadata=True)
    >>> adjacency = graph.adjacency
    >>> embedding = graph.position
    >>> np.round(cosine_modularity(adjacency, embedding), 2)
    0.35
    """
    adjacency = check_format(adjacency)
    total_weight: float = adjacency.data.sum()

    if embedding_col is None:
        check_square(adjacency)
        embedding_col = embedding.copy()

    embedding_row_norm = normalize(embedding, p=2)
    embedding_col_norm = normalize(embedding_col, p=2)

    probs_row = check_probs(weights, adjacency)
    probs_col = check_probs(weights, adjacency.T)

    fit: float = 0.5 * (1 + (np.multiply(embedding_row_norm, adjacency.dot(embedding_col_norm))).sum() / total_weight)
    div: float = 0.5 * (1 + (embedding.T.dot(probs_row)).dot(embedding_col.T.dot(probs_col)))

    if return_all:
        return fit, div, fit - resolution * div
    else:
        return fit - resolution * div
