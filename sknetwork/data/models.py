#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 1, 2019
@author: Quentin Lutz <qlutz@enst.fr>
@author: Nathan de Lara <ndelara@enst.fr>
"""

from typing import Union, Optional

import numpy as np
from scipy import sparse

from sknetwork.utils import Bunch
from sknetwork.utils.format import directed2undirected


def block_model(sizes: np.ndarray, p_in: Union[float, list, np.ndarray] = .2, p_out: float = .05,
                seed: Optional[int] = None, metadata: bool = False) -> Union[sparse.csr_matrix, Bunch]:
    """Stochastic block model.

    Parameters
    ----------
    sizes :
         Block sizes.
    p_in :
        Probability of connection within blocks.
    p_out :
        Probability of connection across blocks (must be less than **p_in**).
    seed : Optional[int]
        Random seed.
    metadata :
        If ``True``, return a `Bunch` object with metadata.

    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Bunch]
        Adjacency matrix or graph with metadata (labels).

    Example
    -------
    >>> from sknetwork.data import block_model
    >>> sizes = np.array([4, 5])
    >>> adjacency = block_model(sizes)
    >>> adjacency.shape
    (9, 9)
    """
    np.random.seed(seed)
    sizes = np.array(sizes)

    if type(p_in) != np.ndarray:
        p_in = p_in * np.ones_like(sizes)
    if np.min(p_in) < p_out:
        raise ValueError('The probability of connection across blocks p_out must be less that the probability of '
                         'connection within a block p_in.')

    # each edge is considered twice
    p_in = p_in / 2
    p_out = p_out / 2

    p_diff = p_in - p_out
    blocks_in = [(sparse.random(s, s, p_diff[k]) > 0) for k, s in enumerate(sizes)]
    adjacency_in = sparse.block_diag(blocks_in)
    n = sizes.sum()
    adjacency_out = sparse.random(n, n, p_out) > 0
    adjacency = sparse.lil_matrix(adjacency_in + adjacency_out)
    adjacency.setdiag(0)
    adjacency = adjacency + adjacency.T
    adjacency = sparse.csr_matrix(adjacency).astype(int)

    if metadata:
        graph = Bunch()
        graph.adjacency = adjacency
        labels = np.repeat(np.arange(len(sizes)), sizes)
        graph.labels = labels
        return graph
    else:
        return adjacency


def erdos_renyie(n: int = 20, p: float = .3, seed: Optional[int] = None) -> sparse.csr_matrix:
    """Erdos-Renyie graph.

    Parameters
    ----------
    n : int
         Number of nodes.
    p : float
        Probability of connection between nodes.
    seed : Optional[int]
        Random seed.

    Returns
    -------
    adjacency : sparse.csr_matrix
        Adjacency matrix.

    Example
    -------
    >>> from sknetwork.data import erdos_renyie
    >>> adjacency = erdos_renyie(7)
    >>> adjacency.shape
    (7, 7)
    """
    return block_model(np.array([n]), p, 0., seed, metadata=False)


def linear_digraph(n: int = 3, metadata: bool = False) -> Union[sparse.csr_matrix, Bunch]:
    """Linear graph (directed).

    Parameters
    ----------
    n : int
        Number of nodes.
    metadata : bool
        If ``True``, return a `Bunch` object with metadata.

    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Bunch]
        Adjacency matrix or graph with metadata (positions).

    Example
    -------
    >>> from sknetwork.data import linear_digraph
    >>> adjacency = linear_digraph(5)
    >>> adjacency.shape
    (5, 5)
    """
    row = np.arange(n - 1)
    col = np.arange(1, n)
    adjacency = sparse.csr_matrix((np.ones(len(row), dtype=int), (row, col)), shape=(n, n))

    if metadata:
        x = np.arange(n)
        y = np.zeros(n)
        graph = Bunch()
        graph.adjacency = adjacency
        graph.position = np.array((x, y)).T
        return graph
    else:
        return adjacency


def linear_graph(n: int = 3, metadata: bool = False) -> Union[sparse.csr_matrix, Bunch]:
    """Linear graph (undirected).

    Parameters
    ----------
    n : int
        Number of nodes.
    metadata : bool
        If ``True``, return a `Bunch` object with metadata.

    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Bunch]
        Adjacency matrix or graph with metadata (positions).

    Example
    -------
    >>> from sknetwork.data import linear_graph
    >>> adjacency = linear_graph(5)
    >>> adjacency.shape
    (5, 5)
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
    n : int
        Number of nodes.
    metadata : bool
        If ``True``, return a `Bunch` object with metadata.

    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Bunch]
        Adjacency matrix or graph with metadata (positions).

    Example
    -------
    >>> from sknetwork.data import cyclic_digraph
    >>> adjacency = cyclic_digraph(5)
    >>> adjacency.shape
    (5, 5)
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
        graph.position = np.array((x, y)).T
        return graph
    else:
        return adjacency


def cyclic_graph(n: int = 3, metadata: bool = False) -> Union[sparse.csr_matrix, Bunch]:
    """Cyclic graph (undirected).

    Parameters
    ----------
    n : int
        Number of nodes.
    metadata : bool
        If ``True``, return a `Bunch` object with metadata.

    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Bunch]
        Adjacency matrix or graph with metadata (positions).

    Example
    -------
    >>> from sknetwork.data import cyclic_graph
    >>> adjacency = cyclic_graph(5)
    >>> adjacency.shape
    (5, 5)
    """
    graph = cyclic_digraph(n, True)
    graph.adjacency = directed2undirected(graph.adjacency)
    if metadata:
        return graph
    else:
        return graph.adjacency
