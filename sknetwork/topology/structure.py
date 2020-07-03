#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 24, 2019
@author: Nathan de Lara <ndelara@enst.fr>
@author: Quentin Lutz <qlutz@enst.fr>
"""
from typing import Tuple, Optional, Union

import numpy as np
from scipy import sparse

from sknetwork.utils.check import is_symmetric, is_square, check_format


def connected_components(adjacency: sparse.csr_matrix, connection: str = 'weak') -> np.ndarray:
    """Extract the connected components of the graph.

    * Graphs
    * Digraphs

    Based on SciPy (scipy.sparse.csgraph.connected_components).

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    connection :
        Must be ``'weak'`` (default) or ``'strong'``. The type of connection to use for directed graphs.

    Returns
    -------
    labels : np.ndarray
        Connected component of each node.
    """
    return sparse.csgraph.connected_components(adjacency, not is_symmetric(adjacency), connection, True)[1]


def largest_connected_component(adjacency: Union[sparse.csr_matrix, np.ndarray], return_labels: bool = False):
    """Extract the largest connected component of a graph. Bipartite graphs are treated as undirected.

    * Graphs
    * Digraphs
    * Bigraphs

    Parameters
    ----------
    adjacency :
        Adjacency or biadjacency matrix of the graph.
    return_labels : bool
        Whether to return the indices of the new nodes in the original graph.

    Returns
    -------
    new_adjacency : sparse.csr_matrix
        Adjacency or biadjacency matrix of the largest connected component.
    indices : array or tuple of array
        Indices of the nodes in the original graph. For biadjacency matrices,
        ``indices[0]`` corresponds to the rows and ``indices[1]`` to the columns.
    """
    adjacency = check_format(adjacency)
    n_row, n_col = adjacency.shape
    if not is_square(adjacency):
        bipartite: bool = True
        full_adjacency = sparse.bmat([[None, adjacency], [adjacency.T, None]], format='csr')
    else:
        bipartite: bool = False
        full_adjacency = adjacency

    labels = connected_components(full_adjacency)
    unique_labels, counts = np.unique(labels, return_counts=True)
    component_label = unique_labels[np.argmax(counts)]
    component_indices = np.where(labels == component_label)[0]

    if bipartite:
        split_ix = np.searchsorted(component_indices, n_row)
        row_ix, col_ix = component_indices[:split_ix], component_indices[split_ix:] - n_row
    else:
        row_ix, col_ix = component_indices, component_indices
    new_adjacency = adjacency[row_ix, :]
    new_adjacency = (new_adjacency.tocsc()[:, col_ix]).tocsr()

    if return_labels:
        if bipartite:
            return new_adjacency, (row_ix, col_ix)
        else:
            return new_adjacency, row_ix
    else:
        return new_adjacency


def is_bipartite(adjacency: sparse.csr_matrix, return_biadjacency: bool = False) \
        -> Union[bool, Tuple[bool, Optional[sparse.csr_matrix], Optional[np.ndarray], Optional[np.ndarray]]]:
    """Check whether an undirected graph is bipartite.

    * Graphs

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


def is_acyclic(adjacency: sparse.csr_matrix) -> bool:
    """Check whether a graph has no cycle.

    Parameters
    ----------
    adjacency:
        Adjacency matrix of the graph.

    Returns
    -------
    is_acyclic : bool
        A boolean with value True if the graph has no cycle and False otherwise
    """
    n_nodes = adjacency.shape[0]
    n_cc = sparse.csgraph.connected_components(adjacency, (not is_symmetric(adjacency)), 'strong', False)
    if n_cc == n_nodes:
        # check for self-loops has they always induce a cycle
        return (adjacency.diagonal() == 0).all()
    else:
        return False
