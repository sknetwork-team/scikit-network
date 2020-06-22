# distutils: language = c++
# cython: language_level=3

"""
Created on June 19, 2020
@author: Pierre Pebereau <pierre.pebereau@telecom-paris.fr>
@author: Alexis Barreaux <alexis.barreaux@telecom-paris.fr>
"""

from typing import Union

import numpy as np
cimport numpy as np
from scipy import sparse

from sknetwork.utils.base import Algorithm


cimport cython

cdef int[:] c_wl_coloring(int[:] indices, int[:] indptr) :
    cdef int n = indptr.shape[0] - 1

    # labels_i denotes the array of the labels at the i-th iteration.
    # labels_i_1 denotes the array of the labels at the i-1-th iteration.
    DTYPE = np.int32
    labels = np.zeros(n, dtype = np.int32)
    cdef int[:]  labels_i = np.ones(n, dtype = DTYPE)
    cdef int[:]  labels_i_1 = np.zeros(n, dtype = DTYPE)

   # cdef int[:]  degres = np.array(indptr[1:]) - np.array(indptr[:-1])

    cdef np.ndarray multiset = np.zeros((n,n), dtype = DTYPE)

    long_label = [ (0,0) for _ in range(n)]


    cdef int i = 1
    cdef int u = 0
    cdef int current_max = 0
    cdef j = 0
    while i < n and (np.array(labels_i_1) != np.array(labels_i)).any() :

        labels_i_1 = np.copy(labels_i) #Perf : ne pas utiliser copy? echanger les addresses ?
        labels_i = np.zeros(n, dtype = DTYPE)

        for v in range(n):
            # 1
            # going through the neighbors of v.
            j = 0
            for u in indices[indptr[v]: indptr[v + 1]]:
                multiset[v][j] = labels_i_1[u]
                j+=1

            # 2
            multiset[v] =  (np.sort(multiset[v]))
            temp_string = str(labels_i_1[v])
            for num in multiset[v] :
                temp_string+=str(num)
            long_label[v] = (temp_string, v) #Efficace ?
        # 3
        long_label.sort(key=lambda x: x[0])  # sort along first axis
        new_hash = {}
        current_max = 0
        for (long_label_v, v) in long_label:
            if not (long_label_v in new_hash):
                new_hash[long_label_v] = current_max
                current_max += 1
            #  4

            labels_i[int(v)] = new_hash[long_label_v]
        i += 1

    labels = labels_i

    return labels






class WLColoring(Algorithm):
    """Weisefeler-Lehman algorithm for coloring/labeling graphs in order to check similarity.

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
    >>> labels = wlcoloring.fit_transform(adjacency)


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

    def __init__(self):
        super(WLColoring, self).__init__()

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

        indices = adjacency.indices
        indptr = adjacency.indptr


        n = indptr.shape[0] - 1

        self.labels_ = np.zeros(n, dtype = np.int32)

        self.labels_ = c_wl_coloring(indices,  indptr)


        return self

    def fit_transform(self, *args, **kwargs) -> np.ndarray:
        """Fit algorithm to the data and return the labels. Same parameters as the ``fit`` method.

        Returns
        -------
        labels : np.ndarray
            Labels.
        """
        self.fit(*args, **kwargs)
        return self.labels_
