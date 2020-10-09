#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on September 17 2020
@author: Tiphaine Viard <tiphaine.viard@telecom-paris.fr>
"""
from collections import deque
from typing import Union
import numpy as np
from scipy import sparse

from sknetwork.ranking.base import BaseRanking
from sknetwork.utils.check import check_format, check_square, check_connected


class Betweenness(BaseRanking):
    """ Betweenness centrality algorithm, from Brandes 2001.

    * Graphs

    Attributes
    ----------
    scores_ : np.ndarray
        Betweenness centrality value of each node

    Example
    -------
    >>> from sknetwork.ranking import Betweenness
    >>> from sknetwork.data.toy_graphs import bow_tie
    >>> betweenness = Betweenness()
    >>> adjacency = bow_tie()
    >>> scores = betweenness.fit_transform(adjacency)
    >>> scores
    array([4., 0., 0., 0., 0.])
    """

    def __init__(self, normalized: bool = False):
        super(Betweenness, self).__init__()
        self.normalized_ = normalized

    def fit(self,
            adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'Betweenness':
        adjacency = check_format(adjacency)
        check_square(adjacency)
        check_connected(adjacency)

        n = adjacency.shape[0]
        self.scores_ = np.zeros(n)
        seen = []  # Using list as stack
        bfs_queue = deque()

        for source in range(n):
            preds = [[] for i in range(n)]
            sigma = n * [0.0]
            sigma[source] = 1.0
            dists = n * [-1]
            dists[source] = 0
            bfs_queue.append(source)

            while len(bfs_queue) != 0:
                i = bfs_queue.popleft()
                seen.append(i)
                neighbors = adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i + 1]]
                for j in neighbors:
                    if dists[j] < 0:  # j found for the first time?
                        dists[j] = dists[i] + 1
                        bfs_queue.append(j)
                    if dists[j] == dists[i] + 1:  # shortest path to j via i?
                        sigma[j] += sigma[i]
                        preds[j].append(i)

            # Now backtrack to compute betweenness scores
            delta = n * [0.0]
            while len(seen) != 0:
                j = seen.pop()
                for i in preds[j]:
                    delta[i] += sigma[i] / sigma[j] * (1 + delta[j])
                if j != source:
                    self.scores_[j] += delta[j]

        # Undirected graph, divide all values by two
        self.scores_ = 1 / 2 * self.scores_

        return self
