"""
Created on June 19, 2020
@author: Pierre Pebereau <pierre.pebereau@telecom-paris.fr>
@author: Alexis Barreaux <alexis.barreaux@telecom-paris.fr>
"""

from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.utils.base import Algorithm


class WLColoring(Algorithm):
    """Weisefeler-Lehman algorithm for coloring/labeling graphs in order to check similarity.

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations.

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
    * Douglas, B. L. (2011).
      'The Weisfeiler-Lehman Method and Graph Isomorphism Testing.
       <https://arxiv.org/pdf/1101.5211.pdf>`_
       Cornell University.


    * Shervashidze, N., Schweitzer, P., van Leeuwen, E. J., Melhorn, K., Borgwardt, K. M. (2010)
      'Weisfeiler-Lehman graph kernels.
      <https://people.mpi-inf.mpg.de/~mehlhorn/ftp/genWLpaper.pdf>`_
      Journal of Machine Learning Research 1, 2010.
    """

    def __init__(self, max_iter=10000):
        super(WLColoring, self).__init__()

        self.max_iter = max_iter
        self.labels_ = None

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

        # labels[0] denotes the array of the labels at the i-th iteration.
        # labels[1] denotes the array of the labels at the i-1-th iteration.
        labels = [[], []]
        labels[1] = np.zeros(n)
        # initializing with the degree of each vertex.
        labels[0] = adjacency.indptr[1:] - adjacency.indptr[:-1]
        i = 1

        while i < self.max_iter and (labels[1] != labels[0]).any():
            multiset = [[] for _ in range(n)]
            labels[1] = np.copy(labels[0])
            long_label = []

            for v in range(n):
                # 1
                # going through the neighbors of v.
                for u in adjacency.indices[adjacency.indptr[v]: adjacency.indptr[v + 1]]:
                    multiset[v].append(labels[1][u])
                # 2
                multiset[v].sort()
                long_label_v = [str(labels[1][v])]
                for value in multiset[v]:
                    long_label_v.append(str(value))
                long_label.append((int("".join(long_label_v)), v))

            # 3

            long_label.sort(key=lambda x: x[0])  # sort along first axis

            new_hash = {}
            current_max = 0
            for (long_label_v, v) in long_label:
                if not (long_label_v in new_hash):
                    new_hash[long_label_v] = current_max
                    current_max += 1
                #  4

                labels[0][int(v)] = new_hash[long_label_v]

            i += 1

        self.labels_ = labels[0]

        return self
