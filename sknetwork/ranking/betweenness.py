#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union
from collections import deque

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from sknetwork.ranking.base import BaseRanking
from sknetwork.utils.check import check_format


class Betweenness(BaseRanking):
    """Betweenness centrality:

    Compute betweenness centrality given adjacency matrix of graph.
    
    Reference
    ---------
    Brandes, U. (2001). `A FasterAlgorithmforBetweennessCentrality 
    <https://www.eecs.wsu.edu/~assefaw/CptS580-06/papers/brandes01centrality.pdf>`_. 
    Journalof Mathematical Sociology 25(2), 163-177.
    """
    def __init__(self):
        super(Betweenness, self).__init__()

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'Betweenness':
        """Betweeness centrality.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.
        
        Returns
        -------
        self: :class:`Betweenness`
        """
        if not isinstance(adjacency, LinearOperator):
            adjacency = check_format(adjacency)
        vertices = np.arange(adjacency._shape[0])
        betweenness = dict((vertex, 0) for vertex in vertices)
        for vertex in vertices:
            stack = []
            P = dict((w, []) for w in vertices)
            sigma = dict((t, 0) for t in vertices); sigma[vertex] = 1
            d = dict((t, -1) for t in vertices); d[vertex] = 0
            queue = deque([])
            queue.append(vertex)
            while queue:
                v = queue.popleft()
                stack.append(v)
                for neighbor in adjacency[v].indices:
                    if d[neighbor] < 0:
                        queue.append(neighbor)
                        d[neighbor] = d[v] + 1
                    if d[neighbor] == d[v] + 1:
                        sigma[neighbor] = sigma[neighbor] + sigma[v]
                        P[neighbor].append(v)
            delta = dict((vertex, 0) for vertex in vertices)
            while stack:
                w = stack.pop()
                for v in P[w]:
                    delta[v] = delta[v] + (sigma[v] / sigma[w]) * (1 + delta[w])
                    if w != vertex:
                        betweenness[w] = betweenness[w] + delta[w]
        self.scores_ = np.array([betweenness[k] for k in sorted(betweenness.keys())])
        return self
