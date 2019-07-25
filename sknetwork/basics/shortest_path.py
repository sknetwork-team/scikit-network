#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 24, 2019
"""

import numpy as np
from scipy import sparse
from typing import Optional


def shortest_path(adjacency: sparse.csr_matrix, method: str = 'auto', directed: bool = True,
                  return_predecessors: bool = False, unweighted: bool = False,
                  overwrite: bool = False, indices: Optional[np.ndarray] = None):
    """Compute the shortest paths in the graph.

    Based on SciPy (scipy.sparse.csgraph.shortest_path)

    Parameters
    ----------
    adjacency:
        The adjacency matrix of the graph
    method:
        The method to be used. Must be ``'auto'`` (default), ``'FW'`` (Floyd-Warshall), ``'D'`` (Dijkstra),
        ``'BF'`` (Bellman-Ford) or ``'J'`` (Johnson).
    directed:
        Denotes if the graph is directed
    return_predecessors:
        If ``True``, the size predecessor matrix is returned
    unweighted:
        If ``True``, the weights of the edges are ignored
    overwrite:
        If ``True`` and ``method == 'FW'``, overwrites the given adjacency
    indices:
        If specified, only compute the paths for the points at the given indices. Will not work with ``method =='FW'``.

    Returns
    -------
    dist_matrix: np.ndarray
        The matrix of distances between graph nodes. ``dist_matrix[i,j]`` gives the shortest
        distance from point ``i`` to point ``j`` along the graph.
    predecessors: np.ndarray
        Returned only if ``return_predecessors == True``. The matrix of predecessors, which can be used to reconstruct
        the shortest paths. Row i of the predecessor matrix contains information on the shortest paths from point ``i``:
        each entry ``predecessors[i, j]`` gives the index of the previous node in the path from point ``i`` to point
        ``j``. If no path exists between point ``i`` and ``j``, then ``predecessors[i, j] = -9999``.
    """
    return sparse.csgraph.shortest_path(adjacency, method, directed,
                                        return_predecessors, unweighted, overwrite, indices)
