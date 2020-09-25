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
    >>> bw = betweenness.fit(adjacency)
    >>> bw.scores_
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
            sigma = [0.0 for i in range(n)]
            sigma[source] = 1.0
            dists = [-1 for i in range(n)]
            dists[source] = 0
            bfs_queue.append(source)

            while len(bfs_queue) != 0:
                v = bfs_queue.popleft()
                seen.append(v)
                neighbours = adjacency.indices[adjacency.indptr[v]:adjacency.indptr[v+1]]
                for w in neighbours:
                    if dists[w] < 0:  # w found for the first time?
                        dists[w] = dists[v] + 1
                        bfs_queue.append(w)
                    if dists[w] == dists[v] + 1:  # shortest path to w via v?
                        sigma[w] = sigma[w] + sigma[v]
                        preds[w].append(v)

            # Now backtrack to compute betweenness scores
            delta = [0.0 for i in range(n)]
            while len(seen) != 0:
                w = seen.pop()
                for v in preds[w]:
                    delta[v] = delta[v] + ((sigma[v]/sigma[w]) * (1 + delta[w]))
                if w != source:
                    self.scores_[w] = self.scores_[w] + delta[w]

        if self.normalized_:
            # Normalize by the max number of (source,target) pairs
            norm_value = 2 / ((n - 1) * (n - 2))
            self.scores_ = self.scores_ * norm_value

        # Undirected graph, divide all values by two
        self.scores_ = 1/2 * self.scores_

        return self
