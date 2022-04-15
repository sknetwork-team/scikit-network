#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 12, 2019
@author: Quentin Lutz <qlutz@enst.fr>
"""
from functools import partial
from multiprocessing import Pool
from typing import Optional, Union, Iterable

import numpy as np
from scipy import sparse

from sknetwork.utils.check import check_n_jobs, is_symmetric


def distance(adjacency: sparse.csr_matrix, sources: Optional[Union[int, Iterable]] = None, method: str = 'D',
             return_predecessors: bool = False, unweighted: bool = False, n_jobs: Optional[int] = None):
    """Compute distances between nodes.

    * Graphs
    * Digraphs

    Based on SciPy (scipy.sparse.csgraph.shortest_path)

    Parameters
    ----------
    adjacency :
        The adjacency matrix of the graph
    sources :
        If specified, only compute the paths for the points at the given indices. Will not work with ``method =='FW'``.
    method :
        The method to be used.

        * ``'D'`` (Dijkstra),
        * ``'BF'`` (Bellman-Ford),
        * ``'J'`` (Johnson).
    return_predecessors :
        If ``True``, the size predecessor matrix is returned
    unweighted :
        If ``True``, the weights of the edges are ignored
    n_jobs :
        If an integer value is given, denotes the number of workers to use (-1 means the maximum number will be used).
        If ``None``, no parallel computations are made.

    Returns
    -------
    dist_matrix : np.ndarray
        The matrix of distances between graph nodes. ``dist_matrix[i,j]`` gives the shortest
        distance from point ``i`` to point ``j`` along the graph.
        If no path exists between nodes ``i`` and ``j``, then ``dist_matrix[i, j] = np.inf``.
    predecessors : np.ndarray, optional
        Returned only if ``return_predecessors == True``. The matrix of predecessors, which can be used to reconstruct
        the shortest paths. Row i of the predecessor matrix contains information on the shortest paths from point ``i``:
        each entry ``predecessors[i, j]`` gives the index of the previous node in the path from point ``i`` to point
        ``j``. If no path exists between nodes ``i`` and ``j``, then ``predecessors[i, j] = -9999``.

    Examples
    --------
    >>> from sknetwork.data import cyclic_digraph
    >>> adjacency = cyclic_digraph(3)
    >>> distance(adjacency, sources=0)
    array([0., 1., 2.])
    >>> distance(adjacency, sources=0, return_predecessors=True)
    (array([0., 1., 2.]), array([-9999,     0,     1]))
    """
    n_jobs = check_n_jobs(n_jobs)
    if method == 'FW' and n_jobs != 1:
        raise ValueError('The Floyd-Warshall algorithm cannot be used with parallel computations.')
    if sources is None:
        sources = np.arange(adjacency.shape[0])
    elif np.issubdtype(type(sources), np.integer):
        sources = np.array([sources])
    n = len(sources)
    directed = not is_symmetric(adjacency)
    local_function = partial(sparse.csgraph.shortest_path,
                             adjacency, method, directed, return_predecessors, unweighted, False)
    if n_jobs == 1 or n == 1:
        try:
            res = sparse.csgraph.shortest_path(adjacency, method, directed, return_predecessors,
                                           unweighted, False, sources)
        except NegativeCycleError as e:
            print("A negative cycle was detected in the graph, and the shortest path could not be returned.")
    else:
        try:
            with Pool(n_jobs) as pool:
                res = np.array(pool.map(local_function, sources))
        except NegativeCycleError as e:
            print("A negative cycle was detected in the graph, and the shortest path could not be returned.")
            pool.terminate()
    if return_predecessors:
        if n == 1:
            return res[0].ravel(), res[1].astype(int).ravel()
        else:
            return res[0], res[1].astype(int)
    else:
        if n == 1:
            return res.ravel()
        else:
            return res


def shortest_path(adjacency: sparse.csr_matrix, sources: Union[int, Iterable], targets: Union[int, Iterable],
                  method: str = 'D', unweighted: bool = False, n_jobs: Optional[int] = None):
    """Compute the shortest paths in the graph.

    * Graphs
    * Digraphs

    Parameters
    ----------
    adjacency :
        The adjacency matrix of the graph
    sources : int or iterable
        Sources nodes.
    targets : int or iterable
        Target nodes.
    method :
        The method to be used.

        * ``'D'`` (Dijkstra),
        * ``'BF'`` (Bellman-Ford),
        * ``'J'`` (Johnson).
    unweighted :
        If ``True``, the weights of the edges are ignored
    n_jobs :
        If an integer value is given, denotes the number of workers to use (-1 means the maximum number will be used).
        If ``None``, no parallel computations are made.

    Returns
    -------
    paths : list
        If single source and single target, return a list containing the nodes on the path from source to target.
        If multiple sources or multiple targets, return a list of paths as lists.
        An empty list means that the path does not exist.

    Examples
    --------
    >>> from sknetwork.data import linear_digraph
    >>> adjacency = linear_digraph(3)
    >>> shortest_path(adjacency, 0, 2)
    [0, 1, 2]
    >>> shortest_path(adjacency, 2, 0)
    []
    >>> shortest_path(adjacency, 0, [1, 2])
    [[0, 1], [0, 1, 2]]
    >>> shortest_path(adjacency, [0, 1], 2)
    [[0, 1, 2], [1, 2]]
    """
    if np.issubdtype(type(sources), np.integer):
        sources = [sources]
    if np.issubdtype(type(targets), np.integer):
        targets = [targets]

    if len(sources) == 1:
        source2target = True
        source = sources[0]
    elif len(targets) == 1:
        source2target = False
        source = targets[0]
        targets = sources
    else:
        raise ValueError(
            'This request is ambiguous. Either use one source and multiple targets or multiple sources and one target.')

    if source2target:
        dists, preds = distance(adjacency, source, method, True, unweighted, n_jobs)
    else:
        dists, preds = distance(adjacency.T, source, method, True, unweighted, n_jobs)

    paths = []
    for target in targets:
        if dists[target] == np.inf:
            path = []
        else:
            path = [target]
            node = target
            while node != source:
                node = preds[node]
                path.append(node)
        if source2target:
            path.reverse()
        paths.append(path)
    if len(paths) == 1:
        paths = paths[0]
    return paths
