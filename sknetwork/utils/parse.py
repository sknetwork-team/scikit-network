#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May, 2020
Nathan de Lara <ndelara@enst.fr>
"""
import numpy as np
from scipy import sparse

from sknetwork.utils.format import directed2undirected


def edgelist2adjacency(edge_list: list, undirected: bool = False) -> sparse.csr_matrix:
    """Build an adjacency matrix from a list of edges.

    Parameters
    ----------
    edge_list : list
        List of edges as pairs (i, j) or triplets (i, j, w) for weighted edges.
    undirected : bool
        If ``True``, return a symmetric adjacency.

    Returns
    -------
    adjacency : sparse.csr_matrix

    Examples
    --------
    >>> edge_list = [(0, 1), (1, 2), (2, 0)]
    >>> adjacency = edgelist2adjacency(edge_list)
    >>> adjacency.shape, adjacency.nnz
    ((3, 3), 3)
    >>> adjacency = edgelist2adjacency(edge_list, undirected=True)
    >>> adjacency.shape, adjacency.nnz
    ((3, 3), 6)
    >>> weighted_edge_list = [(0, 1, 0.2), (1, 2, 4), (2, 0, 1.3)]
    >>> adjacency = edgelist2adjacency(weighted_edge_list)
    >>> adjacency.dtype
    dtype('float64')
    """
    edges = np.array(edge_list)
    row, col = edges[:, 0].astype(np.int32), edges[:, 1].astype(np.int32)
    n = max(row.max(), col.max()) + 1
    if edges.shape[1] > 2:
        data = edges[:, 2]
    else:
        data = np.ones_like(row, dtype=bool)
    adjacency = sparse.csr_matrix((data, (row, col)), shape=(n, n))
    if undirected:
        adjacency = directed2undirected(adjacency)
    return adjacency


def edgelist2biadjacency(edge_list: list) -> sparse.csr_matrix:
    """Build a biadjacency matrix from a list of edges.

    Parameters
    ----------
    edgelist : list
        List of edges as pairs (i, j) or triplets (i, j, w) for weighted edges.

    Returns
    -------
    biadjacency : sparse.csr_matrix

    Examples
    --------
    >>> edge_list = [(0, 0), (1, 0), (1, 1), (2, 1)]
    >>> biadjacency = edgelist2biadjacency(edge_list)
    >>> biadjacency.shape, biadjacency.nnz
    ((3, 2), 4)
    >>> weighted_edge_list = [(0, 0, 0.5), (1, 0, 1), (1, 1, 1), (2, 1, 2)]
    >>> biadjacency = edgelist2biadjacency(weighted_edge_list)
    >>> biadjacency.dtype
    dtype('float64')
    """
    edges = np.array(edge_list)
    row, col = edges[:, 0].astype(np.int32), edges[:, 1].astype(np.int32)
    n_row, n_col = row.max() + 1, col.max() + 1
    if edges.shape[1] > 2:
        data = edges[:, 2]
    else:
        data = np.ones_like(row, dtype=bool)
    biadjacency = sparse.csr_matrix((data, (row, col)), shape=(n_row, n_col))
    return biadjacency
