#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May, 2020
Nathan de Lara <ndelara@enst.fr>
"""
import numpy as np
from scipy import sparse

from sknetwork.utils import directed2undirected


def edgelist2csr(edgelist: list, undirected: bool = False) -> sparse.csr_matrix:
    """Build an adjacency matrix from a list of edges.

    Parameters
    ----------
    edgelist : list
        List of edges as pairs (i, j) or triplets (i, j, w) for weighted edges.
    undirected : bool
        If true, return a symmetric adjacency.

    Returns
    -------
    adjacency : sparse.csr_matrix

    Examples
    --------
    >>> edgelist = [(0, 1), (1, 2), (2, 0)]
    >>> adjacency = edgelist2csr(edgelist)
    >>> adjacency.shape, adjacency.nnz
    ((3, 3), 3)
    >>> adjacency = edgelist2csr(edgelist, undirected=True)
    >>> adjacency.shape, adjacency.nnz
    ((3, 3), 6)
    >>> weighted_edgelist = [(0, 1, 0.2), (1, 2, 4), (2, 0, 1.3)]
    >>> adjacency = edgelist2csr(weighted_edgelist)
    >>> adjacency.dtype
    dtype('float64')
    """
    edges = np.array(edgelist)
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
