#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in May 2023
@author: Thomas Bonald <bonald@enst.fr>
"""
from typing import Iterable, Optional, Union, Tuple

import numpy as np
from scipy import sparse

from sknetwork.utils.format import get_adjacency


def get_distances(input_matrix: sparse.csr_matrix, source: Optional[Union[int, Iterable]] = None,
                  source_row: Optional[Union[int, Iterable]] = None,
                  source_col: Optional[Union[int, Iterable]] = None, transpose: bool = False,
                  force_bipartite: bool = False) \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Get the distances from a source (or a set of sources) in number of hops.

    Parameters
    ----------
    input_matrix :
        Adjacency matrix or biadjacency matrix of the graph.
    source :
        If an integer, index of the source node.
        If a list or array, indices of source nodes (the shortest distances to one of these nodes is returned).
    source_row, source_col :
        For bipartite graphs, index of source nodes on rows and columns.
        The parameter source_row is an alias for source (at least one of them must be ``None``).
    transpose :
        If ``True``, transpose the input matrix.
    force_bipartite :
        If ``True``, consider the input matrix as the biadjacency matrix of a bipartite graph.
        Set to ``True`` is the parameters source_row or source_col re specified.

    Returns
    -------
    distances : np.ndarray of shape (n_nodes,)
        Vector of distances from source (distance = -1 if no path exists from the source).
        For a bipartite graph, two vectors are returned, one for the rows and one for the columns.

    Examples
    --------
    >>> from sknetwork.data import cyclic_digraph
    >>> adjacency = cyclic_digraph(3)
    >>> get_distances(adjacency, source=0)
    array([0, 1, 2])
    >>> get_distances(adjacency, source=[0, 2])
    array([0, 1, 0])
    """
    if transpose:
        matrix = sparse.csr_matrix(input_matrix.T)
    else:
        matrix = input_matrix
    if source_row is not None or source_col is not None:
        force_bipartite = True
    adjacency, bipartite = get_adjacency(matrix, force_bipartite=force_bipartite, allow_empty=True)
    adjacency_transpose = adjacency.astype(bool).T.tocsr()
    n_row, n_col = matrix.shape
    n_nodes = adjacency.shape[0]

    mask = np.zeros(n_nodes, dtype=bool)
    if bipartite:
        if source is not None:
            if source_row is not None:
                raise ValueError('Only one of the parameters source and source_row can be specified.')
            source_row = source
        if source_row is None and source_col is None:
            raise ValueError('At least one of the parameters source_row or source_col must be specified.')
        if source_row is not None:
            mask[source_row] = 1
        if source_col is not None:
            mask[n_row + np.array(source_col)] = 1
    else:
        if source is None:
            raise ValueError('The parameter source must be specified.')
        mask[source] = 1

    distances = -np.ones(n_nodes, dtype=int)
    distances[mask] = 0

    distance = 0
    reach = mask

    while 1:
        distance += 1
        mask = adjacency_transpose.dot(reach).astype(bool) & ~reach
        if np.sum(mask) == 0:
            break
        distances[mask] = distance
        reach |= mask

    if bipartite:
        return distances[:n_row], distances[n_row:]
    else:
        return distances
