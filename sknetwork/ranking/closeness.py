#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 12 2019
@author: Quentin Lutz <qlutz@enst.fr>
"""
from typing import Union, Optional
from scipy import sparse
import numpy as np
from math import log

from sknetwork.utils.algorithm_base_class import Algorithm
from sknetwork.utils.checks import check_format, is_square
from sknetwork.basics import shortest_path


class Closeness(Algorithm):
    """
    Compute the closeness centrality of each node in a connected graph, corresponding to the average length of the
    shortest paths from that node to all the other ones.

    Parameters
    ----------
    method :
        Denotes if the results should be exact or approximate.
    tol:
        If ``method=='approximate'``, the allowed tolerance on each score entry.
    n_jobs:
        If an integer value is given, denotes the number of workers to use (-1 means the maximum number will be used).
        If ``None``, no parallel computations are made.

    Attributes
    ----------
    score_ : np.ndarray
        Closeness centrality of each node.

    Example
    -------
    >>> from sknetwork.toy_graphs import rock_paper_scissors
    >>> closeness = Closeness()
    >>> adjacency = rock_paper_scissors()
    >>> np.round(closeness.fit(adjacency).score_, 2)
    array([0.67, 0.67, 0.67])

    References
    ----------
    Eppstein, D., & Wang, J. (2001, January).
    `Fast approximation of centrality.
    <http://jgaa.info/accepted/2004/EppsteinWang2004.8.1.pdf>`_
    In Proceedings of the twelfth annual ACM-SIAM symposium on Discrete algorithms (pp. 228-229).
    Society for Industrial and Applied Mathematics.
    """
    def __init__(self, method: str = 'exact', tol: float = 1e-1, n_jobs: Optional[int] = None):
        self.method = method
        self.tol = tol
        self.n_jobs = n_jobs
        self.score_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'Closeness':
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

        if self.method == 'exact':
            nb_samples = n
            indices = np.arange(n)
        elif self.method == 'approximate':
            nb_samples = min(int(log(n)/self.tol**2), n)
            indices = np.random.choice(np.arange(n), nb_samples, replace=False)
        else:
            raise ValueError("Method should be either 'exact' or 'approximate'.")

        paths = shortest_path(adjacency, n_jobs=self.n_jobs, indices=indices)

        if paths.max() == np.inf:
            raise ValueError("The graph must be connected.")

        self.score_ = ((n - 1) * nb_samples / n) / paths.T.dot(np.ones(nb_samples))

        return self
