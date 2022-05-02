#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 8, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
from typing import Union, Tuple, Optional

import numpy as np
from scipy import sparse

from sknetwork.linalg.sparse_lowrank import SparseLR
from sknetwork.utils.check import check_format, is_square, is_symmetric
from sknetwork.utils.seeds import stack_seeds, get_seeds


def check_csr_or_slr(adjacency):
    """Check if input is csr or SparseLR and raise an error otherwise."""
    if type(adjacency) not in [sparse.csr_matrix, SparseLR]:
        raise TypeError('Input must be a scipy CSR matrix or a SparseLR object.')
    else:
        return


def directed2undirected(adjacency: Union[sparse.csr_matrix, SparseLR],
                        weighted: bool = True) -> Union[sparse.csr_matrix, SparseLR]:
    """Adjacency matrix of the undirected graph associated with some directed graph.

    The new adjacency matrix becomes either:

    :math:`A+A^T` (default)

    or

    :math:`\\max(A,A^T)`

    If the initial adjacency matrix :math:`A` is binary, bidirectional edges have weight 2
    (first method, default) or 1 (second method).

    Parameters
    ----------
    adjacency :
        Adjacency matrix.
    weighted :
        If ``True``, return the sum of the weights in both directions of each edge.

    Returns
    -------
    new_adjacency :
        New adjacency matrix (same format as input).
    """
    check_csr_or_slr(adjacency)
    if type(adjacency) == sparse.csr_matrix:
        if weighted:
            if adjacency.data.dtype == float:
                data_type = float
            else:
                data_type = int
            new_adjacency = adjacency.astype(data_type)
            new_adjacency += adjacency.T
        else:
            new_adjacency = (adjacency + adjacency.T).astype(bool)
        new_adjacency.tocsr().sort_indices()
        return new_adjacency
    else:
        if weighted:
            new_tuples = [(y, x) for (x, y) in adjacency.low_rank_tuples]
            return SparseLR(directed2undirected(adjacency.sparse_mat), adjacency.low_rank_tuples + new_tuples)
        else:
            raise ValueError('This function only works with ``weighted=True`` for SparseLR objects.')


def bipartite2directed(biadjacency: Union[sparse.csr_matrix, SparseLR]) -> Union[sparse.csr_matrix, SparseLR]:
    """Adjacency matrix of the directed graph associated with a bipartite graph
    (with edges from one part to the other).

    The returned adjacency matrix is:

    :math:`A  = \\begin{bmatrix} 0 & B \\\\ 0 & 0 \\end{bmatrix}`

    where :math:`B` is the biadjacency matrix.

    Parameters
    ----------
    biadjacency :
        Biadjacency matrix of the graph.

    Returns
    -------
    adjacency :
        Adjacency matrix (same format as input).
    """
    check_csr_or_slr(biadjacency)
    n_row, n_col = biadjacency.shape
    if type(biadjacency) == sparse.csr_matrix:
        adjacency = sparse.bmat([[None, biadjacency], [sparse.csr_matrix((n_col, n_row)), None]], format='csr')
        adjacency.sort_indices()
        return adjacency
    else:
        new_tuples = [(np.hstack((x, np.zeros(n_col))), np.hstack((np.zeros(n_row), y)))
                      for (x, y) in biadjacency.low_rank_tuples]
        return SparseLR(bipartite2directed(biadjacency.sparse_mat), new_tuples)


def bipartite2undirected(biadjacency: Union[sparse.csr_matrix, SparseLR]) -> Union[sparse.csr_matrix, SparseLR]:
    """Adjacency matrix of a bigraph defined by its biadjacency matrix.

    The returned adjacency matrix is:

    :math:`A  = \\begin{bmatrix} 0 & B \\\\ B^T & 0 \\end{bmatrix}`

    where :math:`B` is the biadjacency matrix of the bipartite graph.

    Parameters
    ----------
    biadjacency:
        Biadjacency matrix of the graph.

    Returns
    -------
    adjacency :
        Adjacency matrix (same format as input).
    """
    check_csr_or_slr(biadjacency)
    if type(biadjacency) == sparse.csr_matrix:
        adjacency = sparse.bmat([[None, biadjacency], [biadjacency.T, None]], format='csr')
        adjacency.sort_indices()
        return adjacency
    else:
        n_row, n_col = biadjacency.shape
        new_tuples = []
        for (x, y) in biadjacency.low_rank_tuples:
            new_tuples.append((np.hstack((x, np.zeros(n_col))), np.hstack((np.zeros(n_row), y))))
            new_tuples.append((np.hstack((np.zeros(n_row), y)), np.hstack((x, np.zeros(n_col)))))
        return SparseLR(bipartite2undirected(biadjacency.sparse_mat), new_tuples)


def get_adjacency(input_matrix: Union[sparse.csr_matrix, np.ndarray], allow_directed: bool = True,
                  force_bipartite: bool = False, force_directed: bool = False)\
        -> Tuple[sparse.csr_matrix, bool]:
    """Check the input matrix and return a proper adjacency matrix.
    Parameters
    ----------
    input_matrix :
        Adjacency matrix of biadjacency matrix of the graph.
    allow_directed :
        If ``True`` (default), allow the graph to be directed.
    force_bipartite : bool
        If ``True``, return the adjacency matrix of a bipartite graph.
        Otherwise (default), do it only if the input matrix is not square or not symmetric
        with ``allow_directed=False``.
    force_directed :
        If ``True`` return :math:`A  = \\begin{bmatrix} 0 & B \\\\ 0 & 0 \\end{bmatrix}`.
        Otherwise (default), return :math:`A  = \\begin{bmatrix} 0 & B \\\\ B^T & 0 \\end{bmatrix}`.
    """
    bipartite = False
    if force_bipartite or not is_square(input_matrix) or not (allow_directed or is_symmetric(input_matrix)):
        bipartite = True
    if bipartite:
        if force_directed:
            adjacency = bipartite2directed(input_matrix)
        else:
            adjacency = bipartite2undirected(input_matrix)
    else:
        adjacency = input_matrix
    return adjacency, bipartite


def get_adjacency_seeds(input_matrix: Union[sparse.csr_matrix, np.ndarray], allow_directed: bool = True,
                        force_bipartite: bool = False, force_directed: bool = False,
                        seeds: Optional[Union[dict, np.ndarray]] = None,
                        seeds_row: Optional[Union[dict, np.ndarray]] = None,
                        seeds_col: Optional[Union[dict, np.ndarray]] = None,
                        default_value: float = -1,
                        which: Optional[str] = None) \
        -> Tuple[sparse.csr_matrix, np.ndarray, bool]:
    """Check the input matrix and return a proper adjacency matrix with seeds.
    Parameters
    ----------
    input_matrix :
        Adjacency matrix of biadjacency matrix of the graph.
    allow_directed :
        If ``True`` (default), allow the graph to be directed.
    force_bipartite : bool
        If ``True``, return the adjacency matrix of a bipartite graph.
        Otherwise (default), do it only if the input matrix is not square or not symmetric
        with ``allow_directed=False``.
    force_directed :
        If ``True`` return :math:`A  = \\begin{bmatrix} 0 & B \\\\ 0 & 0 \\end{bmatrix}`.
        Otherwise (default), return :math:`A  = \\begin{bmatrix} 0 & B \\\\ B^T & 0 \\end{bmatrix}`.
    seeds :
        Values of seed nodes in initial state (dictionary or vector). Negative values ignored.
    seeds_row, seeds_col :
        Values of rows and columns for bipartite graphs. Negative values ignored.
    default_value :
        Value of non-seed nodes (default = -1).
    which :
        Which seed values.
        If ``'probs'``, return a probability distribution.
        If ``'labels'``, return distinct integer values if all are equal.
    """
    input_matrix = check_format(input_matrix)
    if seeds_row is not None or seeds_col is not None:
        force_bipartite = True
    adjacency, bipartite = get_adjacency(input_matrix, allow_directed=allow_directed,
                                         force_bipartite=force_bipartite, force_directed=force_directed)
    if bipartite:
        if seeds is None:
            seeds = stack_seeds(input_matrix.shape, seeds_row, seeds_col, default_value=default_value)
        else:
            seeds = stack_seeds(input_matrix.shape, seeds, default_value=default_value)
    else:
        seeds = get_seeds(input_matrix.shape, seeds, default_value=default_value)
    if which == 'probs':
        if seeds.sum() > 0:
            seeds /= seeds.sum()
    elif which == 'labels':
        if len(set(seeds[seeds >= 0])) == 1:
            seeds = np.arange(len(seeds))
    return adjacency, seeds, bipartite
