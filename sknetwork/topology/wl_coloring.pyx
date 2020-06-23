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

cdef np.ndarray[long long, ndim=1] c_wl_coloring(int[:] indices, int[:] indptr, int max_iter, np.ndarray[int, ndim=1] input_labels) :
    DTYPE = np.int32
    cdef int n = indptr.shape[0] - 1
    cdef int iteration = 1
    cdef int u = 0
    cdef int j = 0
    cdef int current_max = 0
    cdef int i
    cdef int jj
    cdef int j1
    cdef int j2
    cdef int deg
    cdef int neighbor_label
    cdef long concatenation
    # labels denotes the array of the labels at the i-th iteration.
    # labels_previous denotes the array of the labels at the i-1-th iteration.

    cdef np.ndarray[long long, ndim=1] labels_new
    cdef np.ndarray[long long, ndim=1] labels_old
    cdef np.ndarray[int, ndim = 1]  degres
    cdef np.ndarray [int, ndim = 2] large_label
    cdef np.ndarray [long long, ndim = 1] multiset

    labels_new = np.ones(n, dtype = np.longlong) if input_labels is None else input_labels
    labels_old = np.zeros(n, dtype = np.longlong)
    degres = np.array(indptr[1:]) - np.array(indptr[:-1])
    large_label = np.zeros((n, 2), dtype=DTYPE)

    if max_iter < 0:
        max_iter = n

    while iteration < max_iter and (labels_old != labels_new).any() :
        labels_old = np.copy(labels_new) #Perf : ne pas utiliser copy? echanger les addresses ?


        for i in range(n):
            # 1
            # going through the neighbors of v.
            j = 0
            deg = degres[i]
            multiset = np.empty(deg, dtype=np.longlong)
            j1 = indptr[i]
            j2 = indptr[i + 1]
            for jj in range(j1,j2):
                u = indices[jj]
                multiset[j] = labels_old[u]
                j+=1

            # 2
            multiset =  np.sort(multiset) #np.repeat(np.arange(1+multiset.max()), np.bincount(multiset))
            temp_string = str(labels_old[i])
            j=0

            concatenation = labels_new[i]
            for j in range(deg) :
                neighbor_label = multiset[j]
                concatenation= (concatenation * 10 ** (len(str(neighbor_label)))) + neighbor_label #there are still warnings because of np.int length

            large_label[i] = np.array([concatenation, i])


        # 3

        _, labels_new =  np.unique(large_label[:,0], return_inverse= True)
        iteration += 1
        if iteration == n//4 :
            print("25%")
        if iteration == n//2 :
            print("50%")
        if iteration == 3*n//4 :
            print("75%")
        if iteration == n :
            print("100%")

    return labels_new


class WLColoring(Algorithm):
    """Weisefeler-Lehman algorithm for coloring/labeling graphs in order to check similarity.

    Attributes
    ----------
    labels_ : np.ndarray
        Label of each node.

    max_iter : int
        Maximum iterations of coloring.

    Example
    -------
    >>> from sknetwork.topology import WLColoring
    >>> from sknetwork.data import house
    >>> wlcoloring = WLColoring()
    >>> adjacency = house()
    >>> labels = wlcoloring.fit_transform(adjacency)
    array([1, 2, 0, 0, 2])

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

    def __init__(self, max_iter = -1):
        super(WLColoring, self).__init__()

        self.max_iter = max_iter
        self.labels_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], input_labels : Union[sparse.csr_matrix, np.ndarray] = None) -> 'WLColoring':
        """Fit algorithm to the data.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.

        input_labels :
            Input labels if the user wants to start with a specific input state.

        Returns
        -------
        self: :class:`WLColoring`
        """

        indices = adjacency.indices
        indptr = adjacency.indptr

        self.labels_ = c_wl_coloring(indices, indptr, self.max_iter, input_labels)

        return self

    def fit_transform(self, adjacency: Union[sparse.csr_matrix, np.ndarray], input_labels : Union[sparse.csr_matrix, np.ndarray] = None) -> np.ndarray:
        """Fit algorithm to the data and return the labels. Same parameters as the ``fit`` method.

        Returns
        -------
        labels : np.ndarray
            Labels.
        """
        self.fit(adjacency, input_labels)
        return np.asarray(self.labels_)
