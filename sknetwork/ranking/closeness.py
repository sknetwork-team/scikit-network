#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 12 2019
@author: Quentin Lutz <qlutz@enst.fr>
"""
from math import log
from typing import Union, Optional

import numpy as np
from scipy import sparse

from sknetwork.path.shortest_path import get_distances
from sknetwork.ranking.base import BaseRanking
from sknetwork.utils.check import check_format, check_square, check_connected


class Closeness(BaseRanking):
    """Ranking by closeness centrality of each node in a connected graph, corresponding to the average length of the
    shortest paths from that node to all the other ones.

    Parameters
    ----------
    method :
        Denotes if the results should be exact or approximate.
    tol: float
        If ``method=='approximate'``, the allowed tolerance on each score entry.

    Attributes
    ----------
    scores_ : np.ndarray
        Closeness centrality of each node.

    Example
    -------
    >>> from sknetwork.ranking import Closeness
    >>> from sknetwork.data import cyclic_digraph
    >>> closeness = Closeness()
    >>> adjacency = cyclic_digraph(3)
    >>> scores = closeness.fit_predict(adjacency)
    >>> np.round(scores, 2)
    array([0.67, 0.67, 0.67])

    References
    ----------
    Eppstein, D., & Wang, J. (2001, January).
    `Fast approximation of centrality.
    <http://jgaa.info/accepted/2004/EppsteinWang2004.8.1.pdf>`_
    In Proceedings of the twelfth annual ACM-SIAM symposium on Discrete algorithms (pp. 228-229).
    Society for Industrial and Applied Mathematics.
    """

    def __init__(self, method: str = 'exact', tol: float = 1e-1):
        super(Closeness, self).__init__()

        self.method = method
        self.tol = tol

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'Closeness':
        """Closeness centrality for connected graphs.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`Closeness`
        """
        adjacency = check_format(adjacency)
        check_square(adjacency)
        check_connected(adjacency)
        n = adjacency.shape[0]

        if self.method == 'exact':
            n_sources = n
            sources = np.arange(n)
        elif self.method == 'approximate':
            n_sources = min(int(log(n) / self.tol ** 2), n)
            sources = np.random.choice(np.arange(n), n_sources, replace=False)
        else:
            raise ValueError("Method should be either 'exact' or 'approximate'.")

        distances = np.array([get_distances(adjacency, source=source) for source in sources])

        distances_min = np.min(distances, axis=1)
        scores = (n - 1) / n / np.mean(distances, axis=1)
        scores[distances_min < 0] = 0
        self.scores_ = scores

        return self
