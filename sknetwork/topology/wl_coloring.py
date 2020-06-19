"""
Created on June 19, 2020
@author: Pierre Pebereau <pierre.pebereau@telecom-paris.fr>
@author: Alexis Barreaux <alexis.barreaux@telecom-paris.fr>
"""

from typing import Union
import numpy as np
from scipy import sparse


class WLColoring:
    """Weisefeler-Lehman algorithm for coloring/labeling graphs in order to check similarity.

    * Graphs
    * Digraphs

    Parameters
    ----------
    modularity : str
        Which objective function to maximize. Can be ``'dugue'``, ``'newman'`` or ``'potts'``.
    tol_optimization :
        Minimum increase in the objective function to enter a new optimization pass.


    Attributes
    ----------
    labels_ : np.ndarray
        Label of each node.

    Example
    -------
    >>> from sknetwork.topology import WLColoring
    >>> from sknetwork.data import karate_club
    >>> wlcoloring = WLColoring()
    >>> adjacency = karate_club()
    >>> labels = wlcoloring.fit(adjacency).labels_


    References
    ----------
    >>> https://people.mpi-inf.mpg.de/~mehlhorn/ftp/genWLpaper.pdf

    """

    def __init__(self):
        self.labels_ = None

    def neighborhood(self, adjacency: Union[sparse.csr_matrix, np.ndarray], v: int):
        """Returns the vertices adjacent to vertice v

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.
        v :
            Vertice of which we want to know the neighborhood
        Returns
        -------
        neighbors :
            Array of the neighborhood of v.
        """

        adjacency_array = adjacency.toarray()
        n = adjacency.shape[0]
        neighbors = []

        for u in range(n):
            if adjacency_array[u][v]:
                neighbors.append(u)
        return neighbors

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], max_iter=10000) -> 'WLColoring':
        """Fit algorithm to the data.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.

        max_iter :
            Maximum number of iterations.

        Returns
        -------
        self: :class:`WLColoring`
        """

        n = adjacency.shape[0]

        # Creating the adjacency list.
        neighbors = []
        for v in range(n):
            neighbors.append(self.neighborhood(adjacency, v))

        # Arbitrary choice : initial labelling is size of neighborhood
        # li denotes the array of the labels at the i-th iteration.
        # li_1 denotes the array of the labels at the i-1-th iteration

        li_1 = np.zeros(n)
        li = np.zeros(n)
        i = 1
        li = [len(neighbors[v]) for v in range(n)]

        while i < max_iter and (li_1 != li).any():
            mi = [[] for _ in range(n)]
            li_1 = np.array(li)
            si = []

            for v in range(n):
                # 1
                for u in neighbors[v]:
                    mi[v].append(li_1[u])
                # 2
                mi[v].sort()
                siv = ""
                for value in mi[v]:
                    siv += str(value)
                siv = str(li_1[v]) + siv
                si.append((int(siv), v))

            # 3
            si = np.array(si)
            si.sort(axis=0)  # sort along first axis

            new_hash = {}
            current_max = 0
            for (siv, v) in si:
                if not (siv in new_hash):
                    new_hash[siv] = current_max
                    current_max += 1
                #  4

                li[int(v)] = new_hash[siv]

            i += 1

        self.labels_ = li

        return self
