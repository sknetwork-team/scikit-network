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

    Attributes
    ----------
    labels_ : np.ndarray
        Label of each node.
    max_iter :
        Maximum number of iterations of the algorithm

    Example
    -------
    >>> from sknetwork.topology import WLColoring
    >>> from sknetwork.data import karate_club
    >>> wlcoloring = WLColoring()
    >>> adjacency = karate_club()
    >>> labels = wlcoloring.fit(adjacency).labels_


    References
    ----------
    * Douglas, B. L. (2011).
      'The Weisfeiler-Lehman Method and Graph Isomorphism Testing.
       <https://arxiv.org/pdf/1101.5211.pdf>`_
       Cornell University.

       https://people.mpi-inf.mpg.de/~mehlhorn/ftp/genWLpaper.pdf

    * Shervashidze, N., Schweitzer, P., van Leeuwen, E. J., Melhorn, K., Borgwardt, K. M. (2010)
      'Weisfeiler-Lehman graph kernels.
      <https://people.mpi-inf.mpg.de/~mehlhorn/ftp/genWLpaper.pdf>`_
      Journal of Machine Learning Research 1, 2010.
    """

    def __init__(self, max_iter=10000):
        """Constructor
        Parameters
        ----------
        max_iter :
            Maximum number of iterations.
        """

        self.max_iter = max_iter
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

        neighbors = (adjacency.getrow(v)).nonzero()[1]

        return neighbors

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'WLColoring':
        """Fit algorithm to the data.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.


        Returns
        -------
        self: :class:`WLColoring`
        """

        n = adjacency.shape[0]

        # Creating the adjacency list.
        neighbors = []
        for v in range(n):
            neighbors.append(self.neighborhood(adjacency, v))

        # labels[0] denotes the array of the labels at the i-th iteration.
        # labels[1] denotes the array of the labels at the i-1-th iteration
        labels = [[], []]
        labels[1] = np.zeros(n)
        labels[0] = np.zeros(n)
        i = 1
        labels[0] = [len(neighbors[v]) for v in range(n)]
        while i < self.max_iter and (labels[1] != labels[0]).any():
            multiset = [[] for _ in range(n)]
            labels[1] = np.copy(labels[0])
            si = []

            for v in range(n):
                # 1
                for u in neighbors[v]:
                    multiset[v].append(labels[1][u])
                # 2
                multiset[v].sort()
                siv = [str(labels[1][v])]
                for value in multiset[v]:
                    siv.append(str(value))
                si.append((int("".join(siv)), v))

            # 3

            si.sort(key=lambda x: x[0])  # sort along first axis

            new_hash = {}
            current_max = 0
            for (siv, v) in si:
                if not (siv in new_hash):
                    new_hash[siv] = current_max
                    current_max += 1
                #  4

                labels[0][int(v)] = new_hash[siv]

            i += 1

        self.labels_ = labels[0]

        return self
