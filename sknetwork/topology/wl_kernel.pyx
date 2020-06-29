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
from sknetwork.path.shortest_path import distance

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
    cdef long long[:,:] multiset
    cdef vector[cpair] large_label

    cdef bint has_changed_1 = True
    cdef bint has_changed_2 = True

    max_deg = max(max(list(memoryview(np.array(indptr_1[1:]) - np.array(indptr_1[:n])))), max(list(memoryview(np.array(indptr_2[1:]) - np.array(indptr_2[:n])))))
    has_changed = False
    cdef int[:] count_sort
    cdef int[:] count_1
    cdef int[:] count_2
    cdef long long[:] sorted_multiset = np.empty(max_deg, dtype=np.longlong)

    count_sort= np.zeros(length_count, dtype = DTYPE)
    count_1= np.zeros(length_count, dtype = DTYPE)
    count_2= np.zeros(length_count, dtype = DTYPE)
    multiset = np.empty((n,max_deg), dtype=np.longlong)
    labels_1 = np.ones(n, dtype = np.longlong)
    labels_2 = np.ones(n, dtype = np.longlong)
    large_label = np.zeros((n, 2), dtype=DTYPE)

    if n != m: #graphs with different numbers of nodes can't be similar
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

        labels_1 = c_wl_coloring(indices_1, indptr_1, 1, labels_1, max_deg, n, length_count, new_hash, multiset, sorted_multiset, large_label, count_sort, current_max, False)
        labels_2 = c_wl_coloring(indices_2, indptr_2, 1, labels_2, max_deg, n, length_count, new_hash, multiset, sorted_multiset, large_label, count_sort, current_max, False)
        #print("labels_1", np.asarray(labels_1))
        #print("labels_2", np.asarray(labels_2))
        for i in range(2 * n):
            count_1[i] = 0
            count_2[i] = 0
        for i in range(n):
            count_1[labels_1[i]] += 1
            count_2[labels_2[i]] += 1
        for i in range(2 * n):
            similarity += count_1[i] * count_2[i]
        #print("similarity :", similarity)
        iteration += 1

    return similarity


cdef int c_wl_edge_kernel(int num_iter, np.ndarray[int, ndim=1] indices_1, np.ndarray[int, ndim=1] indptr_1,
                                                 np.ndarray[int, ndim=1] indices_2, np.ndarray[int, ndim=1] indptr_2) :

    cdef int similarity, max_deg1, max_deg2, max_deg, v1, v2, n1, n2, d1, d2, j1, j2, l1_1, l1_2, l2_1, l2_2, iteration, length_count

    n = indptr_1.shape[0] -1
    length_count = 2 * n + 1

    if num_iter < 0 :
        num_iter = n

    cdef int[:]  degrees_1
    cdef int[:]  degrees_2


    cdef long long[:] labels_1
    cdef long long[:] labels_2
    labels_1 = np.ones(n, dtype = np.longlong)
    labels_2 = np.ones(n, dtype = np.longlong)


    degrees_1 = memoryview(np.array(indptr_1[1:]) - np.array(indptr_1[:n]))
    degrees_2 = memoryview(np.array(indptr_2[1:]) - np.array(indptr_2[:n]))
    max_deg1 = max(degrees_1)
    max_deg2 = max(degrees_2)
    max_deg = max(max_deg1, max_deg2)

    cdef np.ndarray[int, ndim = 2] neighbors1 = np.zeros((n, max_deg1),dtype =np.int32)
    cdef np.ndarray[int, ndim = 2] neighbors2 = np.zeros((n, max_deg2),dtype =np.int32)
    cdef np.ndarray[int, ndim = 1] neighborhood


    #Determine adjacency lists
    for v1 in range(n):
        neighborhood = indices_1[indptr_1[v1]: indptr_1[v1+1]]
        for n1 in range(degrees_1[v1]) :
            neighbors1[v1][n1] = neighborhood[n1]

    for v2 in range(n):
        neighborhood = indices_2[indptr_2[v2]: indptr_2[v2+1]]
        for n2 in range(degrees_2[v2]) :
            neighbors2[v2][n2] = neighborhood[n2]

    cdef cmap[long, long] new_hash


    cdef int[:] count_sort
    cdef int[:] count_1
    cdef int[:] count_2
    cdef long long[:] sorted_multiset = np.empty(max_deg, dtype=np.longlong)
    cdef long long[:,:] multiset
    cdef vector[cpair] large_label

    multiset = np.empty((n, max_deg), dtype=np.longlong)
    large_label = np.zeros((n, 2), dtype=np.int32)
    count_sort= np.zeros(length_count, dtype = np.int32)
    count_1= np.zeros(length_count, dtype = np.int32)
    count_2= np.zeros(length_count, dtype = np.int32)

    similarity=0
    for iteration in range(num_iter) :
        new_hash.clear() #une seule fois par itération sur les deux graphes
        current_max = 1

        labels_1 = c_wl_coloring(indices_1, indptr_1, 1, labels_1, max_deg1, n, length_count, new_hash, multiset, sorted_multiset, large_label, count_sort, current_max, False)
        labels_2 = c_wl_coloring(indices_2, indptr_2, 1, labels_2, max_deg2, n, length_count, new_hash, multiset, sorted_multiset, large_label, count_sort, current_max, False)
        #loop on graph 1 edges :
        for v1 in range(n) :
            d1 = degrees_1[v1]
            for j1 in range(d1) :
                n1 = neighbors1[v1][j1]
                if n1 >= v1 : #Proceed in increasing order to ensure each edge is seen exactly once

                    #loop on graph 2 edges :
                    for v2 in range(n) :
                        d2 = degrees_2[v2]
                        for j2 in range(d2) :
                            n2 = neighbors2[v2][j2]
                            if n2 >= v2 :
                                l1_1 = labels_1[v1]
                                l1_2 = labels_1[n1]
                                l2_1 = labels_2[v2]
                                l2_2 = labels_2[n2]

                                if ( l1_1==l2_1  and  l1_2==l2_2 ) or ( l1_2==l2_1  and  l1_1==l2_2 )  : #compare ordered pairs
                                    similarity+=1
    return similarity

cdef int c_wl_shortest_path_kernel(int num_iter, adjacency_1, adjacency_2):

    cdef int similarity, max_deg1, max_deg2, max_deg, v1, v2, d1, d2, j1, j2, l1_1, l1_2, l2_1, l2_2, iteration, length_count

    data_1 = adjacency_1.data
    indices_1 = adjacency_1.indices
    indptr_1 = adjacency_1.indptr

    data_2 = adjacency_2.data
    indices_2 = adjacency_2.indices
    indptr_2 = adjacency_2.indptr



    cdef np.ndarray[double, ndim = 2] dist_1 = distance(adjacency_1)
    cdef np.ndarray[double, ndim = 2] dist_2 = distance(adjacency_2)

    n = indptr_1.shape[0] -1
    length_count = 2 * n + 1

    if num_iter < 0 :
        num_iter = n

    cdef int[:]  degrees_1
    cdef int[:]  degrees_2

    cdef long long[:] labels_1
    cdef long long[:] labels_2
    labels_1 = np.ones(n, dtype = np.longlong)
    labels_2 = np.ones(n, dtype = np.longlong)

    degrees_1 = memoryview(np.array(indptr_1[1:]) - np.array(indptr_1[:n]))
    degrees_2 = memoryview(np.array(indptr_2[1:]) - np.array(indptr_2[:n]))
    max_deg1 = max(degrees_1)
    max_deg2 = max(degrees_2)
    max_deg = max(max_deg1, max_deg2)

    cdef cmap[long, long] new_hash

    cdef int[:] count_sort
    cdef int[:] count_1
    cdef int[:] count_2
    cdef long long[:] sorted_multiset = np.empty(max_deg, dtype=np.longlong)
    cdef long long[:,:] multiset
    cdef vector[cpair] large_label

    multiset = np.empty((n, max_deg), dtype=np.longlong)
    large_label = np.zeros((n, 2), dtype=np.int32)
    count_sort= np.zeros(length_count, dtype = np.int32)
    count_1= np.zeros(length_count, dtype = np.int32)
    count_2= np.zeros(length_count, dtype = np.int32)

    similarity=0

    for iteration in range(num_iter) :
        new_hash.clear() #une seule fois par itération sur les deux graphes
        current_max = 1

        labels_1 = c_wl_coloring(indices_1, indptr_1, 1, labels_1, max_deg1, n, length_count, new_hash, multiset, sorted_multiset, large_label, count_sort, current_max, False)
        labels_2 = c_wl_coloring(indices_2, indptr_2, 1, labels_2, max_deg2, n, length_count, new_hash, multiset, sorted_multiset, large_label, count_sort, current_max, False)
        #loop on graph 1 edges :
        for v1 in range(n) :
            for j1 in range(n) :
                d1 = dist_1[v1][j1]
                if j1 >= v1 and d1 != np.inf: #Proceed in increasing order to ensure each edge is seen exactly once

                    #loop on graph 2 edges :
                    for v2 in range(n) :
                        for j2 in range(n) :
                            d2 = dist_2[v2][j2]
                            if j2 >= v2 and d2 != np.inf :
                                l1_1 = labels_1[v1]
                                l1_2 = labels_1[j1]
                                l2_1 = labels_2[v2]
                                l2_2 = labels_2[j2]

                                if ( l1_1==l2_1  and  l1_2==l2_2 ) or ( l1_2==l2_1  and  l1_1==l2_2 ) and d1 == d2 : #compare ordered pairs
                                    similarity+=1
    return similarity

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
      <http://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf?fbclid=IwAR2l9LJLq2VDfjT4E0ainE2p5dOxtBe89gfSZJoYe4zi5wtuE9RVgzMKmFY>`_
      Journal of Machine Learning Research 1, 2010.
    """

    def __init__(self, max_iter = - 1):
        super(WLKernel, self).__init__()

        self.max_iter = max_iter
        self.labels_ = None

    def fit(self, adjacency_1: Union[sparse.csr_matrix, np.ndarray], adjacency_2: Union[sparse.csr_matrix, np.ndarray], kernel : str = "edge") -> int :
        """Fit algorithm to the data.

        Parameters
        ----------
        adjacency_1 : Union[sparse.csr_matrix, np.ndarray]
            Adjacency matrix of the first graph.

        adjacency_2 : Union[sparse.csr_matrix, np.ndarray]
            Adjacency matrix of the second graph.
        kernel : str
            Type of WL kernel to be used. Types are : subtree, edge, shortest-path


        Returns
        -------
        ret: int
            Likeness between both graphs. -1 if unknown kernel was specified
        """


        ret = -1
        data_1 = adjacency_1.data
        indices_1 = adjacency_1.indices
        indptr_1 = adjacency_1.indptr

        data_2 = adjacency_2.data
        indices_2 = adjacency_2.indices
        indptr_2 = adjacency_2.indptr


        if kernel == "subtree" :
            ret = c_wl_subtree_kernel(self.max_iter,indices_1,  indptr_1,indices_2,  indptr_2)

        if kernel == "edge" :
            ret = c_wl_edge_kernel(self.max_iter,indices_1,  indptr_1,indices_2,  indptr_2)
        if kernel== "shortest path" :

            ret = c_wl_shortest_path_kernel(self.max_iter, adjacency_1, adjacency_2 )
        return ret

    def fit_transform(self, adjacency_g1: Union[sparse.csr_matrix, np.ndarray], adjacency_g2: Union[sparse.csr_matrix, np.ndarray]) -> np.ndarray:
        """Fit algorithm to the data and return the labels. Same parameters as the ``fit`` method.

        Returns
        -------
        labels : np.ndarray
            Labels.
        """
