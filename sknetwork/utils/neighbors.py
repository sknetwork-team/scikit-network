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

    If the graph is directed, returns the vector of successors. Set ``transpose=True``
    to get the predecessors.

    For a biadjacency matrix, returns the neighbors of a row node. Set ``transpose=True``
    to get the neighbors of a column node.

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
        Array of neighbors of the target node.

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


def get_degrees(input_matrix: sparse.csr_matrix, transpose: bool = False) -> np.ndarray:
    """Get the vector of degrees of a graph.

    If the graph is directed, returns the out-degrees (number of successors). Set ``transpose=True``
    to get the in-degrees (number of predecessors).

    For a biadjacency matrix, returns the degrees of rows. Set ``transpose=True``
    to get the degrees of columns.

    Parameters
    ----------
    input_matrix : sparse.csr_matrix
        Adjacency or biadjacency matrix.
    transpose :
        If ``True``, transpose the input matrix.
    Returns
    -------
    degrees : np.ndarray
        Array of degrees.

    Example
    -------
    >>> from sknetwork.data import house
    >>> adjacency = house()
    >>> get_degrees(adjacency)
    array([2, 3, 2, 2, 3], dtype=int32)
    """
    if transpose:
        matrix = sparse.csr_matrix(input_matrix.T)
    else:
        matrix = input_matrix
    degrees = matrix.indptr[1:] - matrix.indptr[:-1]
    return degrees
