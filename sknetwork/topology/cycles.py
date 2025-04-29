#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in February 2024
@author: Thomas Bonald <thomas.bonald@telecom-paris.fr>
@author: Yiwen Peng <yiwen.peng@telecom-paris.fr>
"""
from typing import Tuple, Optional, Union, List

import numpy as np
from scipy import sparse

from sknetwork.utils.check import is_symmetric, check_format
from sknetwork.utils.format import get_adjacency
from sknetwork.path import get_distances


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
        Adjacency matrix of the graph.
    directed :
        Whether to consider the graph as directed (inferred if not specified).

    Returns
    -------
    cycles : list
        List of cycles, each cycle represented as a list of nodes.

    Example
    -------
    >>> from sknetwork.topology import get_cycles
    >>> from sknetwork.data import cyclic_digraph
    >>> graph = cyclic_digraph(4, metadata=True)
    >>> len(get_cycles(graph.adjacency, directed=True))
    1
    """

    if directed is False:
        # the graph must be undirected
        if not is_symmetric(adjacency):
            raise ValueError("The adjacency matrix is not symmetric. The parameter 'directed' must be True.")
    elif directed is None:
        # if not specified, infer from the graph
        directed = not is_symmetric(adjacency)

    cycles = []
    n_nodes = adjacency.shape[0]
    self_loops = np.argwhere(adjacency.diagonal() > 0).ravel()
    if len(self_loops) > 0:
        # add self-loops as cycles
        for node in self_loops:
            cycles.append([node])

    # check if the graph is acyclic
    n_cc, cc_labels = sparse.csgraph.connected_components(adjacency, directed, connection='strong', return_labels=True)
    if directed and n_cc == n_nodes:
        return cycles
    elif not directed:
        # acyclic undirected graph
        n_edges = adjacency.nnz // 2
        if n_cc == n_nodes - n_edges:
            return cycles

    # locate possible cycles
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

    # remove duplicates
    visited_cycles, unique_cycles = set(), []
    for cycle in cycles:
        candidate = np.roll(cycle, -cycle.index(min(cycle)))
        current_cycle = tuple(candidate)
        if not directed:
            current_cycle = tuple(np.sort(candidate))
        if current_cycle not in visited_cycles:
            unique_cycles.append(list(candidate))
        visited_cycles.add(current_cycle)

    return unique_cycles


def break_cycles(adjacency: sparse.csr_matrix, root: Union[int, List[int]],
                 directed: Optional[bool] = None) -> sparse.csr_matrix:
    """Break cycles of a graph from given roots.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    root :
        The root node or list of root nodes to break cycles from.
    directed :
        Whether to consider the graph as directed (inferred if not specified).

    Returns
    -------
    adjacency : sparse.csr_matrix
        Adjacency matrix of the acyclic graph.

    Example
    -------
    >>> from sknetwork.topology import break_cycles, is_acyclic
    >>> from sknetwork.data import cyclic_digraph
    >>> adjacency = cyclic_digraph(4)
    >>> dag = break_cycles(adjacency, root=0, directed=True)
    >>> is_acyclic(dag, directed=True)
    True
    """

    if is_acyclic(adjacency, directed):
        return adjacency

    if root is None:
        raise ValueError("The parameter root must be specified.")
    else:
        out_degree = len(adjacency[root].indices)
        if out_degree == 0:
            raise ValueError("Invalid root node. The root must have at least one outgoing edge.")
        if isinstance(root, int):
            root = [root]

    if directed is False:
        # the graph must be undirected
        if not is_symmetric(adjacency):
            raise ValueError("The adjacency matrix is not symmetric. The parameter 'directed' must be True.")
    elif directed is None:
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
