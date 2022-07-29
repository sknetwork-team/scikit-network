#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in November 2018
@author: Thomas Bonald <thomas.bonald@telecom-paris.fr>
@author: Nathan De Lara <nathan.delara@polytechnique.org>
"""
import numpy as np

from sknetwork.linalg import normalize
from sknetwork.utils.check import check_format, check_square


def get_cosine_similarity(input_matrix, embedding: np.ndarray, embedding_col=None):
    """Average cosine similarity of an embedding :math:`x` defined by:

    :math:`Q = \\sum_{ij}\\dfrac{A_{ij}}{w}\\cos(x_i, x_j)}`

    where :math:`w = 1^TA1` is the total weight of the graph.

    For bipartite graphs with column embedding :math:`y`, the metric is

    :math:`Q = \\sum_{ij} \\dfrac{B_{ij}}{w} \\cos(x_i, y_j)`

    where :math:`w = 1^TB1` is the total weight of the graph.

    Parameters
    ----------
    input_matrix :
        Adjacency matrix or biadjacency matrix of the graph.
    embedding :
        Embedding of the nodes.
    embedding_col :
        Embedding of the columns (for bipartite graphs).

    Returns
    -------
    cosine_similarity : float

    Example
    -------
    >>> from sknetwork.embedding import get_cosine_similarity
    >>> from sknetwork.data import karate_club
    >>> graph = karate_club(metadata=True)
    >>> adjacency = graph.adjacency
    >>> embedding = graph.position
    >>> np.round(get_cosine_similarity(adjacency, embedding), 2)
    0.7
    """
    input_matrix = check_format(input_matrix)
    total_weight = input_matrix.data.sum()

    if embedding_col is None:
        check_square(input_matrix)
        embedding_col = embedding.copy()

    embedding_row_norm = normalize(embedding, p=2)
    embedding_col_norm = normalize(embedding_col, p=2)

    input_matrix_coo = input_matrix.tocoo()
    row = input_matrix_coo.row
    col = input_matrix_coo.col

    cosine_similarity = np.multiply(embedding_row_norm[row], embedding_col_norm[col]).sum() / total_weight

    return cosine_similarity
