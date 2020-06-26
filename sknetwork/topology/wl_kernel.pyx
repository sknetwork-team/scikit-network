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
from sknetwork.topology.wl_coloring cimport c_wl_coloring

from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map as cmap
cimport cython

ctypedef pair[long long, int] cpair
cdef bint compair(pair[long long, int] p1, pair[long long, int] p2):
    return p1.first < p2.first

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int c_wl_subtree_kernel(int num_iter, np.ndarray[int, ndim=1] indices_1, np.ndarray[int, ndim=1] indptr_1,
                             np.ndarray[int, ndim=1] indices_2, np.ndarray[int, ndim=1] indptr_2) :
    DTYPE = np.int32
    cdef int iteration = 1
    cdef int n = indptr_1.shape[0] - 1
    cdef int m = indptr_2.shape[0] - 1
    cdef int length_count = 2 * n + 1
    cdef int i
    cdef int max_deg
    cdef int similarity = 0

    cdef cmap[long, long] new_hash
    cdef long long[:] labels_1
    cdef long long[:] labels_2
    cdef long long[:] multiset
    cdef int[:]  degrees_1
    cdef int[:]  degrees_2
    cdef vector[cpair] large_label

    cdef bint has_changed_1 = True
    cdef bint has_changed_2 = True

    degrees_1 = memoryview(np.array(indptr_1[1:]) - np.array(indptr_1[:n]))
    degrees_2 = memoryview(np.array(indptr_2[1:]) - np.array(indptr_2[:n]))

    max_deg = max(np.max(list(degrees_1)), np.max(list(degrees_2)))
    has_changed = False
    cdef int[:] count_sort
    cdef int[:] count_1
    cdef int[:] count_2
    cdef long long[:] sorted_multiset = np.empty(max_deg, dtype=np.longlong)

    count_sort= np.zeros(length_count, dtype = DTYPE)
    count_1= np.zeros(length_count, dtype = DTYPE)
    count_2= np.zeros(length_count, dtype = DTYPE)
    multiset = np.empty(max_deg, dtype=np.longlong)
    labels_1 = np.ones(n, dtype = np.longlong)
    labels_2 = np.ones(n, dtype = np.longlong)
    large_label = np.zeros((n, 2), dtype=DTYPE)

    if n != m:
        return 0

    if num_iter > 0 :
        num_iter = min(n, num_iter)
    else :
        num_iter = n

    while iteration <= num_iter and (has_changed_1 or has_changed_2) :

        #TODO appel à wl_coloring modifier pour ne faire qu'un tour
        # il faut lui passer tout ce qui est np pour qu'elle n'aie pas à tout redéfinir
        # mettre un paramètre booléen qu'on mettra à true ici

        new_hash.clear() #une seule fois par itération sur les deux graphes
        current_max = 1

        labels_1 = c_wl_coloring(indices_1, indptr_1, 1, labels_1, max_deg, n, length_count, new_hash, degrees_1, multiset, sorted_multiset, large_label, count_sort, current_max, False)
        labels_2 = c_wl_coloring(indices_2, indptr_2, 1, labels_2, max_deg, n, length_count, new_hash, degrees_2, multiset, sorted_multiset, large_label, count_sort, current_max, False)
        print("labels_1", np.asarray(labels_1))
        print("labels_2", np.asarray(labels_2))
        for i in range(2 * n):
            count_1[i] = 0
            count_2[i] = 0
        for i in range(n):
            count_1[labels_1[i]] += 1
            count_2[labels_2[i]] += 1
        for i in range(2 * n):
            similarity += count_1[i] * count_2[i]
        print("similarity :", similarity)
        iteration += 1

    return similarity

cdef int c_wl_edge_kernel(int max_iter, np.ndarray[int, ndim=1] indices_1, np.ndarray[int, ndim=1] indptr_1,
                                                 np.ndarray[int, ndim=1] indices_2, np.ndarray[int, ndim=1] indptr_2) :


    return 0


class WLKernel(Algorithm):
    """Algorithm using Weisefeler-Lehman to check kernels.

    Attributes
    ----------
    labels_ : np.ndarray
        Label of each node.

    max_iter : int
        Maximum number of iterations.

    Example
    -------
    >>> from sknetwork.topology import WLKernel
    >>> from sknetwork.data import house
    >>> wlkernel = WLKernel()
    >>> adjacency = house()
    >>> labels = wlkernel.fit_transform(adjacency)
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

    def __init__(self, max_iter = - 1):
        super(WLKernel, self).__init__()

        self.max_iter = max_iter
        self.labels_ = None

    def fit(self, adjacency_1: Union[sparse.csr_matrix, np.ndarray], adjacency_2: Union[sparse.csr_matrix, np.ndarray]) -> 'WLColoring':
        """Fit algorithm to the data.

        Parameters
        ----------
        adjacency_g1 : Union[sparse.csr_matrix, np.ndarray]
            Adjacency matrix of the first graph.

        adjacency_g2 : Union[sparse.csr_matrix, np.ndarray]
            Adjacency matrix of the second graph.

        Returns
        -------
        ret: int
            Likeness between both graphs.
        """


        indices_1 = adjacency_1.indices
        indptr_1 = adjacency_1.indptr

        indices_2 = adjacency_2.indices
        indptr_2 = adjacency_2.indptr
        ret = c_wl_subtree_kernel(self.max_iter,indices_1,  indptr_1,indices_2,  indptr_2)


        return ret

    def fit_transform(self, adjacency_g1: Union[sparse.csr_matrix, np.ndarray], adjacency_g2: Union[sparse.csr_matrix, np.ndarray]) -> np.ndarray:
        """Fit algorithm to the data and return the labels. Same parameters as the ``fit`` method.

        Returns
        -------
        labels : np.ndarray
            Labels.
        """
