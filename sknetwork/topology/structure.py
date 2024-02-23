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


def is_acyclic(adjacency: sparse.csr_matrix, directed: Optional[bool] = None) -> bool:
    """Check whether a graph has no cycle.

    Parameters
    ----------
    adjacency:
        Adjacency matrix of the graph.
    directed:
        Whether to consider the graph as directed (inferred if not specified).
    Returns
    -------
    is_acyclic : bool
        A boolean with value True if the graph has no cycle and False otherwise.

    Example
    -------
    >>> from sknetwork.topology import is_acyclic
    >>> from sknetwork.data import star, grid
    >>> is_acyclic(star())
    True
    >>> is_acyclic(grid())
    False
    """
    if directed is False:
        # the graph must be undirected
        if not is_symmetric(adjacency):
            raise ValueError("The adjacency matrix is not symmetric. The parameter 'directed' must be True.")
    elif directed is None:
        # if not specified, infer from the graph
        directed = not is_symmetric(adjacency)
    has_loops = (adjacency.diagonal() > 0).any()
    if has_loops:
        return False
    else:
        n_cc = sparse.csgraph.connected_components(adjacency, directed, connection='strong', return_labels=False)
        n_nodes = adjacency.shape[0]
        if directed:
            return n_cc == n_nodes
        else:
            n_edges = adjacency.nnz // 2
            return n_cc == n_nodes - n_edges


def get_cycles(adjacency: sparse.csr_matrix, directed: Optional[bool] = None) -> List[List[int]]:
    """Get all possible cycles of a graph.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the directed graph.
    directed :
        Whether to consider the graph as directed (inferred if not specified).
    
    Returns
    -------
    node_cycles : list
        List of cycles, each cycle represented as a list of nodes index.

    Example
    -------
    >>> from sknetwork.topology import get_cycles
    >>> from sknetwork.data import cyclic_digraph
    >>> graph = cyclic_digraph(4, metadata=True)
    >>> get_cycles(graph.adjacency, directed=True)
    [[0, 1, 2, 3]]
    """

    if directed is False:
        # the graph must be undirected
        if not is_symmetric(adjacency):
            raise ValueError("The adjacency matrix is not symmetric. The parameter 'directed' must be True.")
    elif directed is None:
        # if not specified, infer from the graph
        directed = not is_symmetric(adjacency)

    cycles = []
    num_nodes = adjacency.shape[0]
    self_loops = np.argwhere(adjacency.diagonal() > 0).ravel()
    if len(self_loops) > 0:
        # add self-loops as cycles
        for node in self_loops:
            cycles.append([node])
    
    # check if the graph is acyclic
    n_cc, cc_labels = sparse.csgraph.connected_components(adjacency, directed, connection='strong', return_labels=True)
    if directed and n_cc == num_nodes:
        return cycles
    elif not directed:
        # acyclic undirected graph
        n_edges = adjacency.nnz // 2
        if n_cc == num_nodes - n_edges:
            return cycles
    
    # locate possible cycles position
    labels, counts = np.unique(cc_labels, return_counts=True)
    if directed:
        labels = labels[counts > 1]
    cycle_starts = [np.argwhere(cc_labels == label).ravel()[0] for label in labels]
    
    # find cycles for indicated nodes (depth-first traversal)
    for start_node in cycle_starts:
        stack = [(start_node, [start_node])]
        while stack:
            current_node, path = stack.pop()
            for neighbor in adjacency.indices[adjacency.indptr[current_node]:adjacency.indptr[current_node+1]]:
                if not directed and len(path) > 1 and neighbor == path[-2]: 
                    # ignore the inverse edge (back move) in undirected graph
                    continue
                if neighbor in path:
                    cycles.append(path[path.index(neighbor):])
                else:
                    stack.append((neighbor, path + [neighbor]))

    # post-processing: remove duplicates
    visited_cycles, unique_cycles = set(), []
    for cycle in cycles:
        candidate = np.roll(cycle, -cycle.index(min(cycle)))
        cur_cycle = tuple(candidate)
        if not directed:
            cur_cycle = tuple(np.sort(candidate))
        if cur_cycle not in visited_cycles:
            unique_cycles.append(list(candidate))
        visited_cycles.add(cur_cycle)
        
    return unique_cycles


def break_cycles_from_root(adjacency: sparse.csr_matrix, root: Union[int, List[int]], directed: Optional[bool] = None) -> sparse.csr_matrix:
    """Break cycles from given roots.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the directed graph.
    root :
        The root node or list of root nodes to break cycles from.
    directed :
        Whether to consider the graph as directed.
    
    Returns
    -------
    adjacency : sparse.csr_matrix
        Adjacency matrix of the directed acyclic graph.

    Example
    -------
    >>> from sknetwork.topology import break_cycles_from_root, is_acyclic
    >>> from sknetwork.data import cyclic_digraph
    >>> graph = cyclic_digraph(4, metadata=True)
    >>> graph.adjacency.toarray()
    array([[0, 1, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1],
           [1, 0, 0, 0]])
    >>> dag = break_cycles_from_root(graph.adjacency, root=0, directed=True)
    >>> dag.toarray()
    array([[0, 1, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1],
           [0, 0, 0, 0]])
    >>> is_acyclic(dag, directed=True)
    True
    """

    if is_acyclic(adjacency, directed):
        # raise an exception if the graph is acyclic
        raise ValueError("No cycle found in the graph. The graph is acyclic.")
    
    if root is None:
        raise ValueError("The parameter root must be specified.")
    else:
        out_degrees = adjacency[root, :].sum(axis=1)
        if (out_degrees < 1).all():
            raise ValueError("Unvalid root node. Root must have at least one outgoing edge.")
        if isinstance(root, int):
            root = [root]
    
    if directed is None:
        # if not specified, infer from the graph
        directed = not is_symmetric(adjacency)
    
    # break self-loops
    adjacency = (adjacency - sparse.diags(adjacency.diagonal().astype(int), format='csr')).astype(int)

    if directed:
        # break cycles from the cycle node closest to the root 
        _, cc_labels = sparse.csgraph.connected_components(adjacency, directed, connection='strong', return_labels=True)
        labels, counts = np.unique(cc_labels, return_counts=True)
        cycle_labels = labels[counts > 1]
        distances = get_distances(adjacency, source=root)
        
        for label in cycle_labels:
            cycle_nodes = np.argwhere(cc_labels == label).ravel()
            roots_ix = np.argwhere(distances[cycle_nodes] == min(distances[cycle_nodes])).ravel()
            subroots = set(cycle_nodes[roots_ix])
            stack = [(subroot, [subroot]) for subroot in subroots]
            # break cycles using depth-first traversal
            while stack:
                current_node, path = stack.pop()
                # check if the edge still exists
                if len(path) > 1 and adjacency[path[-2], path[-1]] <= 0:
                    continue
                neighbors = adjacency.indices[adjacency.indptr[current_node]:adjacency.indptr[current_node+1]]
                cycle_neighbors = set(neighbors) & set(cycle_nodes)
                for neighbor in cycle_neighbors:
                    if neighbor in path:
                        adjacency[current_node, neighbor] = 0
                        adjacency.eliminate_zeros()
                    else:
                        stack.append((neighbor, path + [neighbor]))
    else:
        # break cycles from given roots for undirected graphs
        for start_node in root:
            stack = [(start_node, [start_node])]
            # break cycles using depth-first traversal
            while stack:
                current_node, path = stack.pop()
                # check if the edge still exists
                if len(path) > 1 and adjacency[path[-2], path[-1]] <= 0:
                    continue
                neighbors = list(adjacency.indices[adjacency.indptr[current_node]:adjacency.indptr[current_node+1]])
                for neighbor in neighbors:
                    if len(path) > 1 and neighbor == path[-2]:
                        continue
                    if neighbor in path:
                        adjacency[current_node, neighbor] = 0
                        adjacency[neighbor, current_node] = 0
                        adjacency.eliminate_zeros()
                    else:
                        stack.append((neighbor, path + [neighbor]))
    
    return adjacency