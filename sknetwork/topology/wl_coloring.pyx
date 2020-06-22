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
    cdef int i = 1
    cdef int u = 0
    cdef int j = 0
    cdef int current_max = 0

    cdef str temp_string

    # labels denotes the array of the labels at the i-th iteration.
    # labels_previous denotes the array of the labels at the i-1-th iteration.
    DTYPE = np.int32
    cdef int[:] labels = np.ones(n, dtype = DTYPE)
    cdef int[:] labels_previous = np.zeros(n, dtype = DTYPE)

    cdef np.ndarray long_label = np.zeros((n, 2), dtype=DTYPE)
    cdef np.ndarray multiset = np.zeros(n, dtype = DTYPE)

    while i < n and (np.array(labels_previous) != np.array(labels)).any() :
        labels_previous = np.copy(labels) #Perf : ne pas utiliser copy? echanger les addresses ?
        labels = np.zeros(n, dtype = DTYPE)

        for v in range(n):
            # 1
            # going through the neighbors of v.
            j = 0
            for u in indices[indptr[v]: indptr[v + 1]]:
                multiset[j] = labels_previous[u]
                j+=1

            # 2
            multiset =  (np.sort(multiset))
            temp_string = str(labels_previous[v])
            for num in multiset :
                temp_string+=str(num)
            j = int(temp_string)  #j is unused later on and will be reseted
            long_label[v] = np.array([j, v])

            multiset = np.zeros(n, dtype = DTYPE) #reseting multiset
        # 3
        long_label = long_label[long_label[:,0].argsort()]#.sort(key=lambda x: x[0])  # sort along first axis
        new_hash = {}
        current_max = 0

        for j in range(n):
            if not (long_label[j][0] in new_hash):
                new_hash[long_label[j][0]] = current_max
                current_max += 1
            #  4
            labels[long_label[j][1]] = new_hash[long_label[j][0]]
        i += 1

    return labels

cdef int[:,:] counting_sort(n, multiset_v):
    """Sorts an array by using counting sort, variant of bucket sort.

    Parameters
    ----------
    n :
        The size (number of nodes) of the graph.

    multiset_v :
        The array to be sorted.


    Returns
    -------
    sorted_multiset :
        The sorted array.
    """

    cdef int total = 0
    cdef int i
    cdef int[:] count = [0 for _ in range(n)]
    cdef int[:,:] sorted_multiset = [0 for _ in range(len(multiset_v))]

    for i in multiset_v:
        count[i] += 1

    for i in range(n):
        count[i], total = total, count[i] + total

    for i in range(len(multiset_v)):
        sorted_multiset[count[multiset_v[i]]] = multiset_v[i]
        count[multiset_v[i]] += 1

    return sorted_multiset





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
        return np.asarray(self.labels_)
