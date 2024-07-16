#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in July 2019
@author: Thomas Bonald <bonald@enst.fr>
@author: Quentin Lutz <qlutz@enst.fr>
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""
from math import pi
from typing import Union, Optional, Iterable

import numpy as np
from scipy import sparse

from sknetwork.data.base import Dataset
from sknetwork.data.parse import from_edge_list
from sknetwork.utils.check import check_random_state
from sknetwork.utils.format import directed2undirected


def block_model(sizes: Iterable, p_in: Union[float, list, np.ndarray] = .2, p_out: float = .05,
                directed: bool = False, self_loops: bool = False, metadata: bool = False, seed: Optional[int] = None) \
                -> Union[sparse.csr_matrix, Dataset]:
    """Stochastic block model.

    Parameters
    ----------
    sizes :
         Block sizes.
    p_in :
        Probability of connection within blocks.
    p_out :
        Probability of connection across blocks.
    directed :
        If ``True``, return a directed graph.
    self_loops :
         If ``True``, allow self-loops.
    metadata :
        If ``True``, return a `Dataset` object with labels.
    seed :
        Seed of the random generator (optional).
    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Dataset]
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
    random_state = check_random_state(seed)

    if isinstance(p_in, (np.floating, float, int)):
        p_in = p_in * np.ones_like(sizes)
    else:
        p_in = np.array(p_in)

    blocks = []
    for i, a in enumerate(sizes):
        row = []
        for j, b in enumerate(sizes):
            if j == i:
                row.append(sparse.random(a, a, p_in[i], dtype=bool, random_state=random_state))
            else:
                row.append(sparse.random(a, b, p_out, dtype=bool, random_state=random_state))
        blocks.append(row)
    adjacency = sparse.bmat(blocks)
    if not self_loops:
        adjacency = sparse.lil_matrix(adjacency)
        adjacency.setdiag(0)
    if directed:
        adjacency = sparse.csr_matrix(adjacency)
    else:
        adjacency = directed2undirected(sparse.csr_matrix(sparse.triu(adjacency)), weighted=False)
    if metadata:
        graph = Dataset()
        graph.adjacency = adjacency
        labels = np.repeat(np.arange(len(sizes)), sizes)
        graph.labels = labels
        return graph
    else:
        return adjacency


def erdos_renyi(n: int = 20, p: float = .3, directed: bool = False, self_loops: bool = False,
                seed: Optional[int] = None) -> sparse.csr_matrix:
    """Erdos-Renyi graph.

    Parameters
    ----------
    n :
         Number of nodes.
    p :
        Probability of connection between nodes.
    directed :
        If ``True``, return a directed graph.
    self_loops :
         If ``True``, allow self-loops.
    seed :
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
    return block_model([n], p, 0., directed=directed, self_loops=self_loops, metadata=False, seed=seed)


def linear_digraph(n: int = 3, metadata: bool = False) -> Union[sparse.csr_matrix, Dataset]:
    """Linear graph (directed).

    Parameters
    ----------
    n : int
        Number of nodes.
    metadata : bool
        If ``True``, return a `Dataset` object with metadata.

    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Dataset]
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
        graph = Dataset()
        graph.adjacency = adjacency
        graph.position = np.array((x, y)).T
        return graph
    else:
        return adjacency


def linear_graph(n: int = 3, metadata: bool = False) -> Union[sparse.csr_matrix, Dataset]:
    """Linear graph (undirected).

    Parameters
    ----------
    n : int
        Number of nodes.
    metadata : bool
        If ``True``, return a `Dataset` object with metadata.

    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Dataset]
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


def cyclic_digraph(n: int = 3, metadata: bool = False) -> Union[sparse.csr_matrix, Dataset]:
    """Cyclic graph (directed).

    Parameters
    ----------
    n : int
        Number of nodes.
    metadata : bool
        If ``True``, return a `Dataset` object with metadata.

    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Dataset]
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
        graph = Dataset()
        graph.adjacency = adjacency
        graph.position = cyclic_position(n)
        return graph
    else:
        return adjacency


def cyclic_graph(n: int = 3, metadata: bool = False) -> Union[sparse.csr_matrix, Dataset]:
    """Cyclic graph (undirected).

    Parameters
    ----------
    n : int
        Number of nodes.
    metadata : bool
        If ``True``, return a `Dataset` object with metadata.

    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Dataset]
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


def grid(n1: int = 10, n2: int = 10, metadata: bool = False) -> Union[sparse.csr_matrix, Dataset]:
    """Grid (undirected).

    Parameters
    ----------
    n1, n2 : int
        Grid dimension.
    metadata : bool
        If ``True``, return a `Dataset` object with metadata.

    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Dataset]
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
    adjacency = from_edge_list(edges, reindex=False, matrix_only=True)
    if metadata:
        graph = Dataset()
        graph.adjacency = adjacency
        graph.position = np.array(nodes)
        return graph
    else:
        return adjacency


def star(n_branches: int = 3, metadata: bool = False) -> Union[sparse.csr_matrix, Dataset]:
    """Star (undirected).

    Parameters
    ----------
    n_branches : int
        Number of branches.
    metadata : bool
        If ``True``, return a `Dataset` object with metadata (positions).

    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Dataset]
        Adjacency matrix or graph with metadata (positions).

    Example
    -------
    >>> from sknetwork.data import star
    >>> adjacency = star()
    >>> adjacency.shape
    (4, 4)
    """
    edges = [(0, i+1) for i in range(n_branches)]
    adjacency = from_edge_list(edges, reindex=False, matrix_only=True)
    if metadata:
        graph = Dataset()
        graph.adjacency = adjacency
        angles = 2 * np.pi * np.arange(n_branches) / n_branches
        x = [0] + list(np.cos(angles))
        y = [0] + list(np.sin(angles))
        graph.position = np.vstack([x, y]).T
        return graph
    else:
        return adjacency


def albert_barabasi(n: int = 100, degree: int = 3, directed: bool = False, seed: Optional[int] = None) \
        -> sparse.csr_matrix:
    """Albert-Barabasi model.

    Parameters
    ----------
    n : int
        Number of nodes.
    degree : int
        Degree of incoming nodes (less than **n**).
    directed : bool
        If ``True``, return a directed graph.
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
    random_state = check_random_state(seed)
    degrees = np.zeros(n, int)
    degrees[:degree] = degree - 1
    edges = [(i, j) for i in range(degree) for j in range(i)]
    for i in range(degree, n):
        neighbors = random_state.choice(a=i, p=degrees[:i]/degrees.sum(), size=degree, replace=False)
        degrees[neighbors] += 1
        degrees[i] = degree
        edges += [(i, j) for j in neighbors]
    return from_edge_list(edges, directed=directed, reindex=False, matrix_only=True)


def watts_strogatz(n: int = 100, degree: int = 6, prob: float = 0.05, seed: Optional[int] = None,
                   metadata: bool = False) -> Union[sparse.csr_matrix, Dataset]:
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
        If ``True``, return a `Dataset` object with metadata.
    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Dataset]
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
    random_state = check_random_state(seed)
    edges = np.array([(i, (i + j + 1) % n) for i in range(n) for j in range(degree // 2)])
    row, col = edges[:, 0], edges[:, 1]
    adjacency = sparse.coo_matrix((np.ones_like(row, int), (row, col)), shape=(n, n))
    adjacency = sparse.lil_matrix(adjacency + adjacency.T)
    nodes = np.arange(n)
    for i in range(n):
        neighbors = adjacency.rows[i]
        candidates = list(set(nodes) - set(neighbors) - {i})
        for j in neighbors:
            if random_state.random() < prob:
                node = random_state.choice(candidates)
                adjacency[i, node] = 1
                adjacency[node, i] = 1
                adjacency[i, j] = 0
                adjacency[j, i] = 0
    adjacency = sparse.csr_matrix(adjacency, shape=adjacency.shape)
    if metadata:
        graph = Dataset()
        graph.adjacency = adjacency
        graph.position = cyclic_position(n)
        return graph
    else:
        return adjacency
