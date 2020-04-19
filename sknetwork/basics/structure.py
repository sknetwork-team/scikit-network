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


def connected_components(adjacency: sparse.csr_matrix, connection: str = 'weak',
                         return_components: bool = True) -> Union[int, Tuple[int, np.ndarray]]:
    """
    Extract the connected components of the graph

    * Graphs
    * Digraphs

    Based on SciPy (scipy.sparse.csgraph.connected_components).

    Parameters
    ----------
    adjacency:
        Adjacency matrix of the graph.
    connection
        Must be ``'weak'`` (default) or ``'strong'``. The type of connection to use for directed graphs.
    return_components
        If ``True`` (default), then return the labels for each of the connected components.

    Returns
    -------
    n_components: int
        The number of connected components.
    components: ndarray
        The array such that for each node ``i``, ``components[i]`` is the connected component of ``i``.

    """
    return sparse.csgraph.connected_components(adjacency, (not is_symmetric(adjacency)), connection, return_components)


def largest_connected_component(adjacency: Union[sparse.csr_matrix, np.ndarray], return_labels: bool = False):
    """
    Extract the largest connected component of a graph. Bipartite graphs are treated as undirected ones.

    * Graphs
    * Digraphs
    * Bigraphs

    Parameters
    ----------
    adjacency
        Adjacency or biadjacency matrix of the graph.
    return_labels: bool
        Whether to return the indices of the new nodes in the original graph.

    Returns
    -------
    new_adjacency: sparse.csr_matrix
        Adjacency or biadjacency matrix of the largest connected component.
    indices: array or tuple of array
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

    n_components, labels = connected_components(full_adjacency)
    unique_labels, counts = np.unique(labels, return_counts=True)
    component_label = unique_labels[np.argmax(counts)]
    component_indices = np.where(labels == component_label)[0]

    if bipartite:
        split_ix = np.searchsorted(component_indices, n_row)
        samples_ix, features_ix = component_indices[:split_ix], component_indices[split_ix:] - n_row
    else:
        samples_ix, features_ix = component_indices, component_indices
    new_adjacency = adjacency[samples_ix, :]
    new_adjacency = (new_adjacency.tocsc()[:, features_ix]).tocsr()

    if return_labels:
        if bipartite:
            return new_adjacency, (samples_ix, features_ix)
        else:
            return new_adjacency, samples_ix
    else:
        return new_adjacency


def is_bipartite(adjacency: sparse.csr_matrix,
                 return_biadjacency: bool = False) -> Union[bool, Tuple[bool, Optional[sparse.csr_matrix]]]:
    """Check whether an undirected graph is bipartite and can return a possible biadjacency.

    * Graphs

    Parameters
    ----------
    adjacency:
       The symmetric adjacency matrix of the graph.
    return_biadjacency:
        If ``True`` , a possible biadjacency is returned if the graph is bipartite (None is returned otherwise)

    Returns
    -------
    is_bipartite: bool
        A boolean denoting if the graph is bipartite
    biadjacency: sparse.csr_matrix
        A possible biadjacency of the bipartite graph (None if the graph is not bipartite)
    """
    if not is_symmetric(adjacency):
        raise ValueError('The graph must be undirected.')
    if adjacency.diagonal().any():
        if return_biadjacency:
            return False, None
        else:
            return False
    n_nodes = adjacency.indptr.shape[0] - 1
    coloring = np.full(n_nodes, -1, dtype=int)
    exists_remaining = n_nodes
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
                        return False, None
                    else:
                        return False
    if return_biadjacency:
        return True, adjacency[coloring == 0, :][:, coloring == 1]
    else:
        return True
