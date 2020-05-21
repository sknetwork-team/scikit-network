#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 24, 2019
"""
import numpy as np
from scipy import sparse

from sknetwork.utils.check import is_symmetric


def breadth_first_search(adjacency: sparse.csr_matrix, source: int, return_predecessors: bool = True):
    """Breadth-first ordering starting with specified node.

    * Graphs
    * Digraphs

    Based on SciPy (scipy.sparse.csgraph.breadth_first_order)

    Parameters
    ----------
    adjacency :
        The adjacency matrix of the graph
    source : int
        The node from which to start the ordering
    return_predecessors : bool
        If ``True``, the size predecessor matrix is returned

    Returns
    -------
    node_array : np.ndarray
        The breadth-first list of nodes, starting with specified node. The length of node_array is the number of nodes
        reachable from the specified node.
    predecessors : np.ndarray
        Returned only if ``return_predecessors == True``. The list of predecessors of each node in a breadth-first tree.
        If node ``i`` is in the tree, then its parent is given by ``predecessors[i]``. If node ``i`` is not in the tree
        (and for the parent node) then ``predecessors[i] = -9999``.
    """
    directed = not is_symmetric(adjacency)
    return sparse.csgraph.breadth_first_order(adjacency, source, directed, return_predecessors)


def depth_first_search(adjacency: sparse.csr_matrix, source: int, return_predecessors: bool = True):
    """Depth-first ordering starting with specified node.

    * Graphs
    * Digraphs

    Based on SciPy (scipy.sparse.csgraph.depth_first_order)

    Parameters
    ----------
    adjacency :
        The adjacency matrix of the graph
    source :
        The node from which to start the ordering
    return_predecessors:
        If ``True``, the size predecessor matrix is returned

    Returns
    -------
    node_array : np.ndarray
        The depth-first list of nodes, starting with specified node. The length of node_array is the number of nodes
        reachable from the specified node.
    predecessors : np.ndarray
        Returned only if ``return_predecessors == True``. The list of predecessors of each node in a depth-first tree.
        If node ``i`` is in the tree, then its parent is given by ``predecessors[i]``. If node ``i`` is not in the tree
        (and for the parent node) then ``predecessors[i] = -9999``.
    """
    directed = not is_symmetric(adjacency)
    return sparse.csgraph.depth_first_order(adjacency, source, directed, return_predecessors)
