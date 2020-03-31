#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 1, 2019
@author: Quentin Lutz <qlutz@enst.fr>
@author: Nathan de Lara <ndelara@enst.fr>
"""

from typing import Union, Tuple, Optional

import numpy as np
from scipy import sparse

from sknetwork.utils import Bunch
from sknetwork.utils.check import check_is_proba, check_random_state


def block_model(clusters: Union[np.ndarray, int], shape: Optional[Tuple[int, int]] = None, inner_prob: float = .2,
                outer_prob: float = .01, random_state: Optional[Union[np.random.RandomState, int]] = None,
                metadata: bool = False) -> Union[sparse.csr_matrix, Bunch]:
    """Stochastic block model.

    Parameters
    ----------
    clusters : Union[np.ndarray, int]
         Cluster specifications (array of couples where each entry denotes the shape of a cluster
         or an int denoting the number of clusters). If an ``int`` is passed, ``shape`` must be given and
         the clusters are identical in shape.
    shape : Optional[Tuple[int]]
        The size of the adjacency to obtain (might be rectangular for a biadjacency matrix).
    inner_prob : float
        Intra-cluster connection probability.
    outer_prob : float
        Inter-cluster connection probability.
    random_state : Optional[Union[np.random.RandomState, int]]
        Random number generator or random seed. If ``None``, ``numpy.random`` will be used.
    metadata :
        If ``True``, return a `Bunch` object with metadata.

    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Bunch]
        Adjacency matrix or graph with metadata (labels).
    """
    check_is_proba(inner_prob)
    check_is_proba(outer_prob)
    random_state = check_random_state(random_state)

    if type(clusters) == int:
        if not shape:
            raise ValueError('Please specify the shape of the matrix when giving a number of clusters.')
        if clusters <= 0:
            raise ValueError('The number of clusters must be positive.')
        n_clusters = clusters
        clusters_cumul = np.zeros((n_clusters + 1, 2), dtype=int)
        row_step, col_step = shape[0] // clusters, shape[1] // clusters
        if row_step == 0 or col_step == 0:
            raise ValueError('Number of clusters is too high given the shape of the matrix.')
        clusters_cumul[:, 0] = np.arange(0, shape[0] + 1, row_step)
        clusters_cumul[:, 1] = np.arange(0, shape[1] + 1, col_step)
        clusters_cumul[-1, 0] = shape[0]
        clusters_cumul[-1, 1] = shape[1]

    elif type(clusters) == np.ndarray:
        n_clusters = clusters.shape[0]
        clusters_cumul = np.cumsum(clusters, axis=0)
        clusters_cumul = np.insert(clusters_cumul, 0, 0, axis=0)
        if shape:
            if clusters_cumul[-1, 0] != shape[0] or clusters_cumul[-1, 1] != shape[1]:
                raise ValueError('Cluster sizes do not match matrix size.')

    else:
        raise TypeError(
            'Please specify an array of sizes or a number of clusters (along with the shape of the desired matrix).')

    n_row, n_col = clusters_cumul[-1, 0], clusters_cumul[-1, 1]
    labels_row = np.zeros(n_row, dtype=int)
    labels_col = np.zeros(n_col, dtype=int)
    mat = sparse.dok_matrix((n_row, n_col), dtype=bool)
    for label, row_block in enumerate(range(n_clusters)):
        labels_row[clusters_cumul[row_block, 0]:clusters_cumul[row_block + 1, 0]] = label
        labels_col[clusters_cumul[row_block, 1]:clusters_cumul[row_block + 1, 1]] = label
        mask = np.full(n_col, outer_prob)
        mask[clusters_cumul[row_block, 1]:clusters_cumul[row_block + 1, 1]] = inner_prob
        for row in range(clusters_cumul[row_block, 0], clusters_cumul[row_block + 1, 0]):
            mat[row, (random_state.rand(n_col) < mask)] = True
    biadjacency = sparse.csr_matrix(mat)

    if metadata:
        graph = Bunch()
        graph.biadjacency = biadjacency
        graph.labels = labels_row
        graph.labels_row = labels_row
        graph.labels_col = labels_col
        return graph
    else:
        return biadjacency


def linear_digraph(n: int = 3, metadata: bool = False) -> Union[sparse.csr_matrix, Bunch]:
    """Linear graph (directed).

    Parameters
    ----------
    n :
        Number of nodes.
    metadata :
        If ``True``, return a `Bunch` object with metadata.

    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Bunch]
        Adjacency matrix or graph with metadata (positions).

    """
    row = np.arange(n - 1)
    col = np.arange(1, n)
    adjacency = sparse.csr_matrix((np.ones(len(row), dtype=int), (row, col)), shape=(n, n))

    if metadata:
        x = np.arange(n)
        y = np.zeros(n)
        graph = Bunch()
        graph.adjacency = adjacency
        graph.pos = np.array((x, y)).T
        return graph
    else:
        return adjacency


def linear_graph(n: int = 3, metadata: bool = False) -> Union[sparse.csr_matrix, Bunch]:
    """Linear graph (undirected).

    Parameters
    ----------
    n :
        Number of nodes.
    metadata :
        If ``True``, return a `Bunch` object with metadata.

    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Bunch]
        Adjacency matrix or graph with metadata (positions).
    """
    graph = linear_digraph(n, True)
    adjacency = graph.adjacency
    adjacency = adjacency + adjacency.T
    if metadata:
        graph.adjacency = adjacency
        return graph
    else:
        return adjacency


def cyclic_digraph(n: int = 3, metadata: bool = False) -> Union[sparse.csr_matrix, Bunch]:
    """Cyclic graph (directed).

    Parameters
    ----------
    n :
        Number of nodes.
    metadata :
        If ``True``, return a `Bunch` object with metadata.

    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Bunch]
        Adjacency matrix or graph with metadata (positions).

    """
    row = np.arange(n)
    col = np.array(list(np.arange(1, n)) + [0])
    adjacency = sparse.csr_matrix((np.ones(len(row), dtype=int), (row, col)), shape=(n, n))

    if metadata:
        t = 2 * 3.14 * np.arange(n).astype(float) / n
        x = np.cos(t)
        y = np.sin(t)
        graph = Bunch()
        graph.adjacency = adjacency
        graph.pos = np.array((x, y)).T
        return graph
    else:
        return adjacency


def cyclic_graph(n: int = 3, metadata: bool = False) -> Union[sparse.csr_matrix, Bunch]:
    """Cyclic graph (undirected).

    Parameters
    ----------
    n :
        Number of nodes.
    metadata :
        If ``True``, return a `Bunch` object with metadata.

    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Bunch]
        Adjacency matrix or graph with metadata (positions).

    """
    graph = cyclic_digraph(n, True)
    adjacency = graph.adjacency
    adjacency = adjacency + adjacency.T
    if metadata:
        graph.adjacency = adjacency
        return graph
    else:
        return adjacency
