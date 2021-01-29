#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 1, 2019
@author: Thomas Bonald <bonald@enst.fr>
@author: Quentin Lutz <qlutz@enst.fr>
@author: Nathan de Lara <ndelara@enst.fr>
"""
from math import pi
from typing import Union, Optional, Iterable

import numpy as np
from scipy import sparse

from sknetwork.utils import Bunch
from sknetwork.utils.format import directed2undirected
from sknetwork.utils.parse import edgelist2adjacency


def block_model(sizes: Iterable, p_in: Union[float, list, np.ndarray] = .2, p_out: float = .05,
                random_state: Optional[int] = None, metadata: bool = False) \
                -> Union[sparse.csr_matrix, Bunch]:
    """Stochastic block model.

    Parameters
    ----------
    sizes :
         Block sizes.
    p_in :
        Probability of connection within blocks.
    p_out :
        Probability of connection across blocks.
    random_state :
        Seed of the random generator (optional).
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

    References
    ----------
    Airoldi, E.,  Blei, D., Feinberg, S., Xing, E. (2007).
    `Mixed membership stochastic blockmodels. <https://arxiv.org/pdf/0705.4485.pdf>`_
    Journal of Machine Learning Research.
    """
    np.random.seed(random_state)
    sizes = np.array(sizes)

    if isinstance(p_in, (np.floating, float)):
        p_in = p_in * np.ones_like(sizes)
    else:
        p_in = np.array(p_in)

    # each edge is considered twice
    p_in = p_in / 2

    matrix = []
    for i, a in enumerate(sizes):
        row = []
        for j, b in enumerate(sizes):
            if j < i:
                row.append(None)
            elif j > i:
                row.append(sparse.random(a, b, p_out, dtype=bool))
            else:
                row.append(sparse.random(a, a, p_in[i], dtype=bool))
        matrix.append(row)
    adjacency = sparse.bmat(matrix)
    adjacency.setdiag(0)
    adjacency = directed2undirected(adjacency.tocsr(), weighted=False)

    if metadata:
        graph = Bunch()
        graph.adjacency = adjacency
        labels = np.repeat(np.arange(len(sizes)), sizes)
        graph.labels = labels
        return graph
    else:
        return adjacency


def erdos_renyi(n: int = 20, p: float = .3, random_state: Optional[int] = None) -> sparse.csr_matrix:
    """Erdos-Renyi graph.

    Parameters
    ----------
    n :
         Number of nodes.
    p :
        Probability of connection between nodes.
    random_state :
        Seed of the random generator (optional).

    Returns
    -------
    adjacency : sparse.csr_matrix
        Adjacency matrix.

    Example
    -------
    >>> from sknetwork.data import erdos_renyi
    >>> adjacency = erdos_renyi(7)
    >>> adjacency.shape
    (7, 7)

    References
    ----------
    Erdős, P., Rényi, A. (1959). `On Random Graphs. <https://www.renyi.hu/~p_erdos/1959-11.pdf>`_
    Publicationes Mathematicae.
    """
    return block_model(np.array([n]), p, 0., random_state, metadata=False)


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


def cyclic_position(n: int) -> np.ndarray:
    """Position nodes on a circle of unit radius.

    Parameters
    ----------
    n : int
        Number of nodes.

    Returns
    -------
    position : np.ndarray
        Position of nodes.
    """
    t = 2 * pi * np.arange(n).astype(float) / n
    x = np.cos(t)
    y = np.sin(t)
    position = np.array((x, y)).T
    return position


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
        graph = Bunch()
        graph.adjacency = adjacency
        graph.position = cyclic_position(n)
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


def grid(n1: int = 10, n2: int = 10, metadata: bool = False) -> Union[sparse.csr_matrix, Bunch]:
    """Grid (undirected).

    Parameters
    ----------
    n1, n2 : int
        Grid dimension.
    metadata : bool
        If ``True``, return a `Bunch` object with metadata.

    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Bunch]
        Adjacency matrix or graph with metadata (positions).

    Example
    -------
    >>> from sknetwork.data import grid
    >>> adjacency = grid(10, 5)
    >>> adjacency.shape
    (50, 50)
    """
    nodes = [(i1, i2) for i1 in range(n1) for i2 in range(n2)]
    edges = [((i1, i2), (i1 + 1, i2)) for i1 in range(n1 - 1) for i2 in range(n2)]
    edges += [((i1, i2), (i1, i2 + 1)) for i1 in range(n1) for i2 in range(n2 - 1)]
    node_id = {u: i for i, u in enumerate(nodes)}
    edges = list(map(lambda edge: (node_id[edge[0]], node_id[edge[1]]), edges))
    adjacency = edgelist2adjacency(edges, undirected=True)
    if metadata:
        graph = Bunch()
        graph.adjacency = adjacency
        graph.position = np.array(nodes)
        return graph
    else:
        return adjacency


def albert_barabasi(n: int = 100, degree: int = 3, undirected: bool = True, seed: Optional[int] = None) \
        -> sparse.csr_matrix:
    """Albert-Barabasi model.

    Parameters
    ----------
    n : int
        Number of nodes.
    degree : int
        Degree of incoming nodes (less than **n**).
    undirected : bool
        If ``True``, return an undirected graph.
    seed :
        Seed of the random generator (optional).

    Returns
    -------
    adjacency : sparse.csr_matrix
        Adjacency matrix.

    Example
    -------
    >>> from sknetwork.data import albert_barabasi
    >>> adjacency = albert_barabasi(30, 3)
    >>> adjacency.shape
    (30, 30)

    References
    ----------
    Albert, R., Barabási, L. (2002). `Statistical mechanics of complex networks
    <https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.74.47>`_
    Reviews of Modern Physics.
    """
    np.random.seed(seed)
    degrees = np.zeros(n, int)
    degrees[:degree] = degree - 1
    edges = [(i, j) for i in range(degree) for j in range(i)]
    for i in range(degree, n):
        neighbors = np.random.choice(i, p=degrees[:i]/degrees.sum(), size=degree, replace=False)
        degrees[neighbors] += 1
        degrees[i] = degree
        edges += [(i, j) for j in neighbors]
    return edgelist2adjacency(edges, undirected)


def watts_strogatz(n: int = 100, degree: int = 6, prob: float = 0.05, seed: Optional[int] = None,
                   metadata: bool = False) -> Union[sparse.csr_matrix, Bunch]:
    """Watts-Strogatz model.

    Parameters
    ----------
    n :
        Number of nodes.
    degree :
        Initial degree of nodes.
    prob :
        Probability of edge modification.
    seed :
        Seed of the random generator (optional).
    metadata :
        If ``True``, return a `Bunch` object with metadata.
    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Bunch]
        Adjacency matrix or graph with metadata (positions).

    Example
    -------
    >>> from sknetwork.data import watts_strogatz
    >>> adjacency = watts_strogatz(30, 4, 0.02)
    >>> adjacency.shape
    (30, 30)

    References
    ----------
    Watts, D., Strogatz, S. (1998). Collective dynamics of small-world networks, Nature.
    """
    np.random.seed(seed)
    edges = np.array([(i, (i + j + 1) % n) for i in range(n) for j in range(degree // 2)])
    row, col = edges[:, 0], edges[:, 1]
    adjacency = sparse.coo_matrix((np.ones_like(row, int), (row, col)), shape=(n, n))
    adjacency = sparse.lil_matrix(adjacency + adjacency.T)
    nodes = np.arange(n)
    for i in range(n):
        neighbors = adjacency.rows[i]
        candidates = list(set(nodes) - set(neighbors) - {i})
        for j in neighbors:
            if np.random.random() < prob:
                node = np.random.choice(candidates)
                adjacency[i, node] = 1
                adjacency[node, i] = 1
                adjacency[i, j] = 0
                adjacency[j, i] = 0
    adjacency = sparse.csr_matrix(adjacency, shape=adjacency.shape)
    if metadata:
        graph = Bunch()
        graph.adjacency = adjacency
        graph.position = cyclic_position(n)
        return graph
    else:
        return adjacency
