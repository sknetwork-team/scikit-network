#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in July 2019
@author: Nathan de Lara <nathan.delara@polytechnique.org>
@author: Quentin Lutz <qlutz@enst.fr>
@author: Thomas Bonald <tbonald@enst.fr>
"""
from typing import Tuple, Optional, Union, List

import numpy as np
from scipy import sparse

from sknetwork.utils.check import is_symmetric, check_format
from sknetwork.utils.format import get_adjacency
from sknetwork.path import get_distances


def get_connected_components(input_matrix: sparse.csr_matrix, connection: str = 'weak', force_bipartite: bool = False) \
        -> np.ndarray:
    """Extract the connected components of a graph.

    Parameters
    ----------
    input_matrix :
        Input matrix (either the adjacency matrix or the biadjacency matrix of the graph).
    connection :
        Must be ``'weak'`` (default) or ``'strong'``. The type of connection to use for directed graphs.
    force_bipartite : bool
        If ``True``, consider the input matrix as the biadjacency matrix of a bipartite graph.

    Returns
    -------
    labels :
        Connected component of each node.
        For bipartite graphs, rows and columns are concatenated (rows first).

    Example
    -------
    >>> from sknetwork.topology import get_connected_components
    >>> from sknetwork.data import house
    >>> get_connected_components(house())
    array([0, 0, 0, 0, 0], dtype=int32)
    """
    input_matrix = check_format(input_matrix)
    if len(input_matrix.data) == 0:
        raise ValueError('The graph is empty (no edge).')
    adjacency, _ = get_adjacency(input_matrix, force_bipartite=force_bipartite)
    labels = sparse.csgraph.connected_components(adjacency, connection=connection, return_labels=True)[1]
    return labels


def is_connected(input_matrix: sparse.csr_matrix, connection: str = 'weak', force_bipartite: bool = False) -> bool:
    """Check whether the graph is connected.

    Parameters
    ----------
    input_matrix :
        Input matrix (either the adjacency matrix or the biadjacency matrix of the graph).
    connection :
        Must be ``'weak'`` (default) or ``'strong'``. The type of connection to use for directed graphs.
    force_bipartite : bool
        If ``True``, consider the input matrix as the biadjacency matrix of a bipartite graph.

    Example
    -------
    >>> from sknetwork.topology import is_connected
    >>> from sknetwork.data import house
    >>> is_connected(house())
    True
    """
    return len(set(get_connected_components(input_matrix, connection, force_bipartite))) == 1


def get_largest_connected_component(input_matrix: sparse.csr_matrix, connection: str = "weak",
                                    force_bipartite: bool = False, return_index: bool = False) \
        -> Union[sparse.csr_matrix, Tuple[sparse.csr_matrix, np.ndarray]]:
    """Extract the largest connected component of a graph. Bipartite graphs are treated as undirected.

    Parameters
    ----------
    input_matrix :
        Adjacency matrix or biadjacency matrix of the graph.
    connection :
        Must be ``'weak'`` (default) or ``'strong'``. The type of connection to use for directed graphs.
    force_bipartite : bool
        If ``True``, consider the input matrix as the biadjacency matrix of a bipartite graph.
    return_index : bool
        Whether to return the index of the nodes of the largest connected component in the original graph.

    Returns
    -------
    output_matrix : sparse.csr_matrix
        Adjacency matrix or biadjacency matrix of the largest connected component.
    index : array
        Indices of the nodes in the original graph.
        For bipartite graphs, rows and columns are concatenated (rows first).

    Example
    -------
    >>> from sknetwork.topology import get_largest_connected_component
    >>> from sknetwork.data import house
    >>> get_largest_connected_component(house()).shape
    (5, 5)
    """
    input_matrix = check_format(input_matrix)
    adjacency, bipartite = get_adjacency(input_matrix, force_bipartite=force_bipartite)
    labels = get_connected_components(adjacency, connection=connection)
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_component_label = unique_labels[np.argmax(counts)]

    if bipartite:
        n_row, n_col = input_matrix.shape
        index_row = np.argwhere(labels[:n_row] == largest_component_label).ravel()
        index_col = np.argwhere(labels[n_row:] == largest_component_label).ravel()
        index = np.hstack((index_row, index_col))
        output_matrix = input_matrix[index_row, :]
        output_matrix = (output_matrix.tocsc()[:, index_col]).tocsr()
    else:
        index = np.argwhere(labels == largest_component_label).ravel()
        output_matrix = input_matrix[index, :]
        output_matrix = (output_matrix.tocsc()[:, index]).tocsr()
    if return_index:
        return output_matrix, index
    else:
        return output_matrix


def is_bipartite(adjacency: sparse.csr_matrix, return_biadjacency: bool = False) \
        -> Union[bool, Tuple[bool, Optional[sparse.csr_matrix], Optional[np.ndarray], Optional[np.ndarray]]]:
    """Check whether a graph is bipartite.

    Parameters
    ----------
    adjacency :
       Adjacency matrix of the graph (symmetric).
    return_biadjacency :
        If ``True``, return a biadjacency matrix of the graph if bipartite.

    Returns
    -------
    is_bipartite : bool
        A boolean denoting if the graph is bipartite.
    biadjacency : sparse.csr_matrix
        A biadjacency matrix of the graph if bipartite (optional).
    rows : np.ndarray
        Index of rows in the original graph (optional).
    cols : np.ndarray
        Index of columns in the original graph (optional).

    Example
    -------
    >>> from sknetwork.topology import is_bipartite
    >>> from sknetwork.data import cyclic_graph
    >>> is_bipartite(cyclic_graph(4))
    True
    >>> is_bipartite(cyclic_graph(3))
    False
    """
    if not is_symmetric(adjacency):
        raise ValueError('The graph must be undirected.')
    if adjacency.diagonal().any():
        if return_biadjacency:
            return False, None, None, None
        else:
            return False
    n = adjacency.indptr.shape[0] - 1
    coloring = np.full(n, -1, dtype=int)
    exists_remaining = n
    while exists_remaining:
        src = np.argwhere(coloring == -1)[0, 0]
        next_nodes = [src]
        coloring[src] = 0
        exists_remaining -= 1
        while next_nodes:
            node = next_nodes.pop()
            for neighbor in adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node + 1]]:
                if coloring[neighbor] == -1:
                    coloring[neighbor] = 1 - coloring[node]
                    next_nodes.append(neighbor)
                    exists_remaining -= 1
                elif coloring[neighbor] == coloring[node]:
                    if return_biadjacency:
                        return False, None, None, None
                    else:
                        return False
    if return_biadjacency:
        rows = np.argwhere(coloring == 0).ravel()
        cols = np.argwhere(coloring == 1).ravel()
        return True, adjacency[rows, :][:, cols], rows, cols
    else:
        return True


