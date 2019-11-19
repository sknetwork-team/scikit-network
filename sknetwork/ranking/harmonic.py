#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 19 2019
@author: Quentin Lutz <qlutz@enst.fr>
"""
from typing import Union, Optional
from scipy import sparse
import numpy as np

from sknetwork.utils.algorithm_base_class import Algorithm
from sknetwork.utils.checks import check_format, is_square
from sknetwork.basics import shortest_path, is_connected


class Harmonic(Algorithm):
    """
    Compute the harmonic centrality of each node in a connected graph, corresponding to the average inverse length of
    the shortest paths from that node to all the other ones.

    For a directed graph, the harmonic centrality is computed in terms of outgoing paths.

    Parameters
    ----------
    n_jobs:
        If an integer value is given, denotes the number of workers to use (-1 means the maximum number will be used).
        If ``None``, no parallel computations are made.

    Attributes
    ----------
    score_ : np.ndarray
        Harmonic centrality of each node.

    Example
    -------
    >>> harmonic = Harmonic()
    >>> adjacency = sparse.identity(3).tocsr()
    >>> np.round(harmonic.fit(adjacency).score_, 2)
    array([0., 0., 0.])

    References
    ----------
    Marchiori, M., & Latora, V. (2000).
    `Harmony in the small-world.
    <https://arxiv.org/pdf/cond-mat/0008357.pdf>`_
    Physica A: Statistical Mechanics and its Applications, 285(3-4), 539-546.
    """
    def __init__(self, n_jobs: Optional[int] = None):
        self.n_jobs = n_jobs
        self.score_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'Harmonic':
        """
        Closeness centrality for connected graphs.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`Closeness`
        """
        adjacency = check_format(adjacency)
        n = adjacency.shape[0]
        if not is_square(adjacency):
            raise ValueError("The adjacency is not square. Please use 'bipartite2undirected',"
                             "'bipartite2directed' or 'BiCloseness'.")

        indices = np.arange(n)

        paths = shortest_path(adjacency, n_jobs=self.n_jobs, indices=indices)

        inv = (1 / paths)
        np.fill_diagonal(inv, 0)

        self.score_ = inv.dot(np.ones(n))

        return self
