#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 19 2019
@author: Quentin Lutz <qlutz@enst.fr>
"""
from typing import Union, Optional

import numpy as np
from scipy import sparse

from sknetwork.path.shortest_path import get_distances
from sknetwork.ranking.base import BaseRanking
from sknetwork.utils.check import check_format, check_square


class Harmonic(BaseRanking):
    """Harmonic centrality of each node in a connected graph, corresponding to the average inverse length of
    the shortest paths from that node to all the other ones.

    For a directed graph, the harmonic centrality is computed in terms of outgoing paths.

    Parameters
    ----------
    n_jobs:
        If an integer value is given, denotes the number of workers to use (-1 means the maximum number will be used).
        If ``None``, no parallel computations are made.

    Attributes
    ----------
    scores_ : np.ndarray
        Score of each node.

    Example
    -------
    >>> from sknetwork.ranking import Harmonic
    >>> from sknetwork.data import house
    >>> harmonic = Harmonic()
    >>> adjacency = house()
    >>> scores = harmonic.fit_transform(adjacency)
    >>> np.round(scores, 2)
    array([3. , 3.5, 3. , 3. , 3.5])

    References
    ----------
    Marchiori, M., & Latora, V. (2000).
    `Harmony in the small-world.
    <https://arxiv.org/pdf/cond-mat/0008357.pdf>`_
    Physica A: Statistical Mechanics and its Applications, 285(3-4), 539-546.
    """

    def __init__(self, n_jobs: Optional[int] = None):
        super(Harmonic, self).__init__()

        self.n_jobs = n_jobs

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'Harmonic':
        """Harmonic centrality for connected graphs.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`Harmonic`
        """
        adjacency = check_format(adjacency)
        check_square(adjacency)
        n = adjacency.shape[0]
        indices = np.arange(n)

        dists = get_distances(adjacency, n_jobs=self.n_jobs, sources=indices)

        np.fill_diagonal(dists, 1)
        inv = (1 / dists)
        np.fill_diagonal(inv, 0)

        self.scores_ = inv.dot(np.ones(n))

        return self
