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
        n, m = biadjacency.shape
        new_tuples = []
        for (x, y) in biadjacency.low_rank_tuples:
            new_tuples.append((np.hstack((x, np.zeros(m))), np.hstack((np.zeros(n), y))))
            new_tuples.append((np.hstack((np.zeros(n), y)), np.hstack((x, np.zeros(m)))))
        return SparseLR(bipartite2undirected(biadjacency.sparse_mat), new_tuples)
    else:
        raise TypeError('Input must be a scipy CSR matrix or a SparseLR object.')


def set_adjacency(adjacency: sparse.csr_matrix, force_undirected: bool, force_biadjacency: bool) -> sparse.csr_matrix:
    """Transform adjacency to match algorithms requirements.

    Parameters
    ----------
    adjacency:
        Input matrix.
    force_undirected:
        If the adjacency is not already symmetric and force_biadjacency==False, returns A+A^T.
    force_biadjacency:
        If True, returns the mirror adjacency even for square matrices.

    Returns
    -------
        The modified adjacency.

    """

    n1, n2 = adjacency.shape

    if n1 != n2 or force_biadjacency:
        # bipartite graph
        if force_undirected:
            adjacency = bipartite2undirected(adjacency)
        else:
            adjacency = bipartite2directed(adjacency)
    else:
        # non-bipartite graph
        if force_undirected and not is_symmetric(adjacency):
            adjacency = directed2undirected(adjacency)

    return adjacency


def set_adjacency_weights(adjacency: sparse.csr_matrix, weights: Union[str, np.ndarray],
                          secondary_weights: Union[None, str, np.ndarray], force_undirected: bool,
                          force_biadjacency: bool) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Modify adjacency and compute weights.

    Parameters
    ----------
    adjacency
    weights
    secondary_weights
    force_undirected
    force_biadjacency

    Returns
    -------
        adjacency, weights, secondary_weights

    """

    n1, n2 = adjacency.shape

    if n1 != n2 or force_biadjacency:
        # bipartite graph
        if force_undirected:
            adjacency = bipartite2undirected(adjacency)
            out_weights = check_probs(weights, adjacency)
            in_weights = out_weights
        else:
            if secondary_weights is None:
                if type(weights) == str:
                    secondary_weights = weights
                else:
                    warnings.warn(Warning("Feature_weights have been set to 'degree'."))
                    secondary_weights = 'degree'
            out_weights = np.hstack((check_probs(weights, adjacency), np.zeros(n2)))
            in_weights = np.hstack((np.zeros(n1), check_probs(secondary_weights, adjacency.T)))
            adjacency = bipartite2directed(adjacency)
    else:
        # non-bipartite graph
        if force_undirected and not is_symmetric(adjacency):
            adjacency = directed2undirected(adjacency)
        out_weights = check_probs(weights, adjacency)
        in_weights = check_probs(weights, adjacency.T)

    return adjacency, out_weights, in_weights
