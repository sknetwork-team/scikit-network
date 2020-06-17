from collections import deque
from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.ranking.base import BaseRanking
from sknetwork.utils.check import check_format


class Betweenness(BaseRanking):
    """Betweenness centrality:

    Examples
    --------
    >>> from sknetwork.data.toy_graphs import house
    >>> adjacency = house()
    >>> betweeness = Betweenness()
    >>> scores = betweeness.fit_transform(adjacency)
    array([0., 3., 1., 1., 3.])

    References
    ----------
    Brandes, U. (2001). `A FasterAlgorithmforBetweennessCentrality 
    <https://www.eecs.wsu.edu/~assefaw/CptS580-06/papers/brandes01centrality.pdf>`_. 
    Journalof Mathematical Sociology 25(2), 163-177.
    """
    def __init__(self):
        super(Betweenness, self).__init__()

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'Betweenness':
        self.scores_ = _brandes(adjacency)
        return self


def _brandes(A):
    """compute betweenness centrality given adjacency matrix of graph.

    https://www.eecs.wsu.edu/~assefaw/CptS580-06/papers/brandes01centrality.pdf
    """
    V = np.arange(A._shape[0])
    
    C = dict((v, 0) for v in V)
    for s in V:
        S = []
        P = dict((w, []) for w in V)
        g = dict((t, 0) for t in V); g[s] = 1
        d = dict((t, -1) for t in V); d[s] = 0
        Q = deque([])
        Q.append(s)
        while Q:
            v = Q.popleft()
            S.append(v)
            for w in A[v].indices:
                if d[w] < 0:
                    Q.append(w)
                    d[w] = d[v] + 1
                if d[w] == d[v] + 1:
                    g[w] = g[w] + g[v]
                    P[w].append(v)
        e = dict((v, 0) for v in V)
        while S:
            w = S.pop()
            for v in P[w]:
                e[v] = e[v] + (g[v] / g[w]) * (1 + e[w])
                if w != s:
                    C[w] = C[w] + e[w]
    return np.array([C[k] for k in sorted(C.keys())])