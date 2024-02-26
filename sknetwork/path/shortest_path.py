#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in May 2023
@author: Thomas Bonald <bonald@enst.fr>
"""
from typing import Iterable, Optional, Union, Tuple

import numpy as np
from scipy import sparse

from sknetwork.path.dag import get_dag
from sknetwork.utils.format import bipartite2undirected
from sknetwork.path.distances import get_distances


def get_shortest_path(input_matrix: sparse.csr_matrix, source: Optional[Union[int, Iterable]] = None,
                      source_row: Optional[Union[int, Iterable]] = None,
                      source_col: Optional[Union[int, Iterable]] = None, force_bipartite: bool = False) \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Get the shortest paths from a source (or a set of sources) in number of hops.

    Parameters
    ----------
    input_matrix :
        Adjacency matrix or biadjacency matrix of the graph.
    source :
        If an integer, index of the source node.
        If a list, indices of source nodes (the shortest distances to one of these nodes in returned).
    source_row, source_col :
        For bipartite graphs, index of source nodes on rows and columns.
        The parameter source_row is an alias for source (at least one of them must be ``None``).
    force_bipartite :
        If ``True``, consider the input matrix as the biadjacency matrix of a bipartite graph.
        Set to ``True`` is the parameters source_row or source_col are specified.

    Returns
    -------
    path : sparse.csr_matrix
        Adjacency matrix of the graph of the shortest paths from the source node (or the set of source nodes).
        If the input graph is a bipartite graph, the shape of the matrix is (n_row + n_col, n_row + n_col) with the new
        index corresponding to the rows then the columns of the original graph.

    Examples
    --------
    >>> from sknetwork.data import cyclic_digraph
    >>> adjacency = cyclic_digraph(3)
    >>> path = get_shortest_path(adjacency, source=0)
    >>> path.toarray().astype(int)
    array([[0, 1, 0],
           [0, 0, 1],
           [0, 0, 0]])
    """
    distances = get_distances(input_matrix, source, source_row, source_col, force_bipartite)
    if type(distances) == tuple:
        adjacency = bipartite2undirected(input_matrix)
        distances = np.hstack(distances)
    else:
        adjacency = input_matrix
    return get_dag(adjacency, order=distances)

