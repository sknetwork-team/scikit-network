#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 8, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import warnings
from typing import Union, Tuple

import numpy as np
from scipy import sparse

from sknetwork.linalg.sparse_lowrank import SparseLR
from sknetwork.utils.checks import check_probs, is_symmetric


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
            raise ValueError('This function only works with ``weight_sum=True`` for SparseLR objects.')
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
    n1, n2 = biadjacency.shape
    if type(biadjacency) == sparse.csr_matrix:
        return sparse.bmat([[None, biadjacency], [sparse.csr_matrix((n2, n1)), None]], format='csr')
    elif type(biadjacency) == SparseLR:
        new_tuples = [(np.hstack((x, np.zeros(n2))), np.hstack((np.zeros(n1), y)))
                      for (x, y) in biadjacency.low_rank_tuples]
        return SparseLR(bipartite2directed(biadjacency.sparse_mat), new_tuples)
    else:
        raise TypeError('Input must be a scipy CSR matrix or a SparseLR object.')


def bipartite2undirected(biadjacency: Union[sparse.csr_matrix, SparseLR]) -> Union[sparse.csr_matrix, SparseLR]:
    """
    Returns the adjacency matrix of a biadjacency adjacency defined by its biadjacency matrix.

    The returned adjacency matrix is:

    :math:`A  = \\begin{bmatrix} 0 & B \\\\ B^T & 0 \\end{bmatrix}`

    where :math:`B` is the biadjacency matrix of the bipartite graph.

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
        n1, n2 = biadjacency.shape
        new_tuples = []
        for (x, y) in biadjacency.low_rank_tuples:
            new_tuples.append((np.hstack((x, np.zeros(n2))), np.hstack((np.zeros(n1), y))))
            new_tuples.append((np.hstack((np.zeros(n1), y)), np.hstack((x, np.zeros(n2)))))
        return SparseLR(bipartite2undirected(biadjacency.sparse_mat), new_tuples)
    else:
        raise TypeError('Input must be a scipy CSR matrix or a SparseLR object.')


def set_adjacency_weights(adjacency: sparse.csr_matrix, weights: Union[str, np.ndarray],
                          col_weights: Union[None, str, np.ndarray], force_undirected: bool,
                          force_biadjacency: bool) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Modify adjacency and compute weights.

    * :math:`\\begin{bmatrix} 0 & A \\\\ 0 & 0 \\end{bmatrix}` if :math:`A` is not square or **force_biadjacency**
    is ``True``

    * :math:`\\begin{bmatrix} 0 & A \\\\ A^T & 0 \\end{bmatrix}` if in addition, **force_undirected** is ``True``

    * :math:`A + A^T` if :math:`A` is square, **force_biadjacency** is ``False`` and **force_undirected** is ``True``

    * :math:`A` otherwise

    Parameters
    ----------
    adjacency :
        Adjacency or biadjacency matrix of the graph.
    weights :
        Weights of nodes.
    col_weights :
        Weights of secondary nodes (for bipartite graphs).
    force_undirected :
        If ``True``, consider the graph as undirected.
    force_biadjacency :
        If ``True``, consider the input matrix as a biadjacency matrix.

    Returns
    -------
        adjacency, weights, col_weights : Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]

    """

    n1, n2 = adjacency.shape

    if n1 != n2 or force_biadjacency:
        # bipartite graph
        if force_undirected:
            adjacency = bipartite2undirected(adjacency)
            weights = check_probs(weights, adjacency)
            col_weights = weights
        else:
            if col_weights is None:
                if type(weights) == str:
                    col_weights = weights
                else:
                    warnings.warn(Warning("Feature_weights have been set to 'degree'."))
                    col_weights = 'degree'
            weights = np.hstack((check_probs(weights, adjacency), np.zeros(n2)))
            col_weights = np.hstack((np.zeros(n1), check_probs(col_weights, adjacency.T)))
            adjacency = bipartite2directed(adjacency)
    else:
        # non-bipartite graph
        if force_undirected and not is_symmetric(adjacency):
            adjacency = directed2undirected(adjacency)
        weights = check_probs(weights, adjacency)
        col_weights = check_probs(weights, adjacency.T)

    return adjacency, weights, col_weights
