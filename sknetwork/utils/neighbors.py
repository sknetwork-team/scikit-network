#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 29, 2020
@author: Thomas Bonald <bonald@enst.fr>
"""
import numpy as np
from scipy import sparse


def get_neighbors(adjacency: sparse.csr_matrix, node: int) -> np.ndarray:
    """.

    Parameters
    ----------
    adjacency : sparse.csr_matrix
        Adjacency or biadjacency matrix.
    node : int
        Target node.

    Returns
    -------
    neighbors : np.ndarray
        Array of neighbors of the target node (successors if the graph is directed).

    Example
    -------
    >>> adjacency = sparse.csr_matrix([[1, 1, 0], [0, 1, 0], [1, 1, 0]])
    >>> get_neighbors(adjacency, 0)
    array([0, 1], dtype=int32)
    """
    neighbors = adjacency.indices[adjacency.indptr[node]: adjacency.indptr[node + 1]]
    return neighbors
