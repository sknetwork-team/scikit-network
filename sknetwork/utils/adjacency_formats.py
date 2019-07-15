#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 8, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import numpy as np
from scipy import sparse
from sknetwork.linalg.sparse_lowrank import SparseLR
from typing import Union


def directed2undirected(adjacency: Union[sparse.csr_matrix, SparseLR],
                        weight_sum: bool = True) -> Union[sparse.csr_matrix, SparseLR]:
    """
    Returns the adjacency matrix of the undirected graph associated with some directed graph.

    The new adjacency matrix becomes either:

    :math:`A+A^T` (default)

    or

    :math:`\\max(A,A^T)`

    If the initial adjacency matrix :math:`A` is binary, bidirectional edges have weight 2
    (first method, default) or 1 (second method).

    Parameters
    ----------
    adjacency:
        Adjacency matrix.
    weight_sum:
        If True, return the sum of the weights in both directions of each edge.

    Returns
    -------
    New adjacency matrix (symmetric).
    """
    if type(adjacency) == sparse.csr_matrix:
        if weight_sum:
            return sparse.csr_matrix(adjacency + adjacency.T)
        else:
            return adjacency.maximum(adjacency.T)
    elif type(adjacency) == SparseLR:
        if weight_sum:
            new_tuples = [(y, x) for (x, y) in adjacency.low_rank_tuples]
            return SparseLR(directed2undirected(adjacency.sparse_mat), adjacency.low_rank_tuples + new_tuples)
        else:
            raise ValueError('This function only works with ``weight_sum==True`` for SparseLR objects.')
    else:
        raise TypeError('Input must be a scipy CSR matrix or a SparseLR object.')


def bipartite2directed(biadjacency: Union[sparse.csr_matrix, SparseLR]) -> Union[sparse.csr_matrix, SparseLR]:
    """
    Returns the adjacency matrix of the directed graph associated with a bipartite graph
    (with edges from one part to the other).

    The returned adjacency matrix is:

    :math:`A  = \\begin{bmatrix} 0 & B \\\\ 0 & 0 \\end{bmatrix}`

    where :math:`B` is the biadjacency matrix.

    Parameters
    ----------
    biadjacency:
        Biadjacency matrix of the graph.

    Returns
    -------
    Adjacency matrix.
    """
    n, m = biadjacency.shape
    if type(biadjacency) == sparse.csr_matrix:
        return sparse.bmat([[None, biadjacency], [sparse.csr_matrix((m, n)), None]], format='csr')
    elif type(biadjacency) == SparseLR:
        new_tuples = [(np.hstack((x, np.zeros(m))), np.hstack((np.zeros(n), y)))
                      for (x, y) in biadjacency.low_rank_tuples]
        return SparseLR(bipartite2directed(biadjacency.sparse_mat), new_tuples)
    else:
        raise TypeError('Input must be a scipy CSR matrix or a SparseLR object.')


def bipartite2undirected(biadjacency: Union[sparse.csr_matrix, SparseLR]) -> Union[sparse.csr_matrix, SparseLR]:
    """
    Returns the adjacency matrix of a biadjacency adjacency defined by its biadjacency matrix.

    The returned adjacency matrix is:

    :math:`A  = \\begin{bmatrix} 0 & B \\\\ B^T & 0 \\end{bmatrix}`

    where :math:`B` is the biadjacency matrix of the biadjacency adjacency.

    Parameters
    ----------
    biadjacency:
        Biadjacency matrix of the graph.

    Returns
    -------
    Adjacency matrix (symmetric).
    """
    if type(biadjacency) == sparse.csr_matrix:
        return sparse.bmat([[None, biadjacency], [biadjacency.T, None]], format='csr')
    elif type(biadjacency) == SparseLR:
        n, m = biadjacency.shape
        new_tuples = []
        for (x, y) in biadjacency.low_rank_tuples:
            new_tuples.append((np.hstack((x, np.zeros(m))), np.hstack((np.zeros(n), y))))
            new_tuples.append((np.hstack((np.zeros(n), y)), np.hstack((x, np.zeros(m)))))
        return SparseLR(bipartite2undirected(biadjacency.sparse_mat), new_tuples)
    else:
        raise TypeError('Input must be a scipy CSR matrix or a SparseLR object.')
