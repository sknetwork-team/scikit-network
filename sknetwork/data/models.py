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


def block_model(sizes: np.ndarray, p_in: Union[float, np.ndarray] = .2, p_out: float = .05,
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
        graph.position = np.array((x, y)).T
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
        graph.position = np.array((x, y)).T
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
