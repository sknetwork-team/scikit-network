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
from sknetwork.topology.wl_coloring cimport c_wl_coloring, c_wl_coloring_2
from sknetwork.path.shortest_path import distance

from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map as cmap
cimport cython

ctypedef pair[long long, int] cpair

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int initialise_kernel(int num_iter,
                           adjacency_1: Union[sparse.csr_matrix, np.ndarray],
                           adjacency_2: Union[sparse.csr_matrix, np.ndarray],
                           str kernel_type):

    cdef np.ndarray[int, ndim=1] indices_1 = adjacency_1.indices
    cdef np.ndarray[int, ndim=1] indptr_1 = adjacency_1.indptr
    cdef np.ndarray[int, ndim=1] indices_2 = adjacency_2.indices
    cdef np.ndarray[int, ndim=1] indptr_2 = adjacency_2.indptr
    #TODO those two are only used for shortest path, find smth to do ?
    cdef np.ndarray[double, ndim = 2] dist_1 = distance(adjacency_1)
    cdef np.ndarray[double, ndim = 2] dist_2 = distance(adjacency_2)

    cdef int iteration = 0
    cdef int n = indptr_1.shape[0] - 1
    cdef int m = indptr_2.shape[0] - 1

    if n != m: #TODO changer ça
        return 0

    cdef int length_count = 2 * n + 1
    cdef int i
    cdef int similarity = 0
    cdef int current_max
    cdef bint has_changed_1
    cdef bint has_changed_2
    cdef bint res #only for isomorphism

    cdef long long[:] labels_1
    cdef long long[:] labels_2
    cdef long long[:,:] multiset

    cdef vector[cpair] large_label

    cdef int[:] count_1
    cdef int[:] count_2

    cdef cmap[long long, long long] new_hash

    max_deg = max(np.max(indptr_1[1:] - indptr_1[:n]), np.max(indptr_2[1:] - indptr_2[:n]))

    count_1= np.zeros(length_count, dtype = np.int32)
    count_2= np.zeros(length_count, dtype = np.int32)
    multiset = np.empty((n,max_deg), dtype=np.longlong)
    labels_1 = np.ones(n, dtype = np.longlong)
    labels_2 = np.ones(n, dtype = np.longlong)
    large_label = np.zeros((n, 2), dtype= np.int32)


    if num_iter > 0 :
        num_iter = min(n, num_iter)
    else :
        num_iter = n

    if kernel_type == "isomorphism":
        similarity = 1

    while iteration < num_iter : #and (has_changed_1 or has_changed_2), not using this atm cause it gives issues when
                                #not normalizing
        new_hash.clear()
        new_hash, current_max , _ = c_wl_coloring(indices_1, indptr_1, 1, labels_1, multiset, large_label, 1, new_hash, False)
        _ ,current_max, _ = c_wl_coloring(indices_2, indptr_2, 1, labels_2, multiset, large_label, current_max, new_hash, False)
        iteration += 1
        if kernel_type == "isomorphism":
            if c_wl_isomorphism(n, count_1, count_2, labels_1, labels_2) == 0:
                return 0
            #TODO oui c'est moche, est ce qu'on peut faire mieux ?
            continue

        if kernel_type == "subtree":
            similarity = c_wl_subtree_kernel(n, count_1, count_2, labels_1, labels_2, similarity)
            continue

        if kernel_type == "edge":
            similarity = c_wl_edge_kernel(n, indices_1, indptr_1, indices_2, indptr_2, labels_1, labels_2, similarity)
            continue

        if kernel_type == "path":
            similarity = c_wl_shortest_path_kernel(n, dist_1, dist_2, labels_1, labels_2, similarity)
            continue

    return similarity

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int c_wl_isomorphism(int n,
                          int[:] count_1,
                          int[:] count_2,
                          long long[:] labels_1,
                          long long[:] labels_2):

    for i in range(2 * n + 1):
        count_1[i] = 0
        count_2[i] = 0
    for i in range(n):
        count_1[labels_1[i]] += 1
        count_2[labels_2[i]] += 1
    for i in range(2 * n + 1):
        if count_1[i] != count_2[i]:
            return 0

    return 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int c_wl_subtree_kernel(int n,
                             int[:] count_1,
                             int[:] count_2,
                             long long[:] labels_1,
                             long long[:] labels_2,
                             similarity):

    for i in range(2 * n + 1):
        count_1[i] = 0
        count_2[i] = 0
    for i in range(n):
        count_1[labels_1[i]] += 1
        count_2[labels_2[i]] += 1
    for i in range(2 * n + 1):
        similarity += count_1[i] * count_2[i]

    return similarity

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int c_wl_edge_kernel(int n,
                          np.ndarray[int, ndim=1] indices_1,
                          np.ndarray[int, ndim=1] indptr_1,
                          np.ndarray[int, ndim=1] indices_2,
                          np.ndarray[int, ndim=1] indptr_2,
                          long long[:] labels_1,
                          long long[:] labels_2,
                          similarity):


    cdef int v1, v2, n1, n2, j1, j2, jj1, jj2, d1, d2, l1_1, l1_2, l2_1, l2_2
    #TODO on perd un peu de temps à faire ça mais ça rend les choses jolies

    #loop on graph 1 edges :
    for v1 in range(n) :
        j1 = indptr_1[v1]
        j2 = indptr_1[v1+1]
        for d1 in range(j2 -j1) :
            n1 = indices_1[d1 + j1]
            if n1 >= v1 : #Proceed in increasing order to ensure each edge is seen exactly once

                #loop on graph 2 edges :
                for v2 in range(n) :
                    jj1 = indptr_2[v2]
                    jj2 = indptr_2[v2 + 1]
                    for d2 in range(jj2 -jj1) :
                        n2 = indices_2[d2 + jj1]
                        if n2 >= v2 :
                            l1_1 = labels_1[v1]
                            l1_2 = labels_1[n1]
                            l2_1 = labels_2[v2]
                            l2_2 = labels_2[n2]

                            if ( l1_1==l2_1  and  l1_2==l2_2 ) or ( l1_2==l2_1  and  l1_1==l2_2 )  : #compare ordered pairs
                                similarity+=1
    return similarity

@cython.boundscheck(False)
@cython.wraparound(False)
#TODO shortest path is currently wrong.
cdef int c_wl_shortest_path_kernel(int n,
                                   np.ndarray[double, ndim = 2] dist_1,
                                   np.ndarray[double, ndim = 2] dist_2,
                                   long long[:] labels_1,
                                   long long[:] labels_2,
                                   int similarity):

    cdef int  v1, v2, d1, d2, j1, j2, l1_1, l1_2, l2_1, l2_2
    #TODO on perd un peu de temps à faire ça mais ça rend les choses jolies

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
    #TODO change this example
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

    def fit(self, adjacency_1: Union[sparse.csr_matrix, np.ndarray], adjacency_2: Union[sparse.csr_matrix, np.ndarray], kernel : str = "subtree") -> int :
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
        #TODO checker les entrées
        ret = -1

        ret = initialise_kernel(self.max_iter, adjacency_1, adjacency_2, kernel)

        return ret
