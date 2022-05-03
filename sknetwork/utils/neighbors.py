#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 29, 2020
@author: Thomas Bonald <bonald@enst.fr>
"""
import numpy as np
from scipy import sparse


def get_neighbors(input_matrix: sparse.csr_matrix, node: int, transpose: bool = False) -> np.ndarray:
    """Get the neighbors of a node.

    Parameters
    ----------
    input_matrix : sparse.csr_matrix
        Adjacency or biadjacency matrix.
    node : int
        Target node.
    transpose :
        If ``True``, transpose the input matrix.
    Returns
    -------
    neighbors : np.ndarray
        Array of neighbors of the target node (successors for directed graphs; set ``transpose=True`` for predecessors).

    Example
    -------
    >>> from sknetwork.data import house
    >>> adjacency = house()
    >>> get_neighbors(adjacency, node=0)
    array([1, 4], dtype=int32)
    """
    if transpose:
        matrix = sparse.csr_matrix(input_matrix.T)
    else:
        matrix = input_matrix
    neighbors = matrix.indices[matrix.indptr[node]: matrix.indptr[node + 1]]
    return neighbors
