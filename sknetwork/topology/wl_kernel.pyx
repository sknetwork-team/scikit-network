# distutils: language = c++
# cython: language_level=3

"""
Created on July 1, 2020
@author: Pierre Pebereau <pierre.pebereau@telecom-paris.fr>
@author: Alexis Barreaux <alexis.barreaux@telecom-paris.fr>
"""

from typing import Union

import numpy as np
cimport numpy as np
from scipy import sparse

from sknetwork.topology.wl_coloring cimport c_wl_coloring

from libcpp.pair cimport pair
cimport cython

ctypedef pair[long long, int] cpair

def wl_kernel(adjacency_1: Union[sparse.csr_matrix, np.ndarray],
              adjacency_2: Union[sparse.csr_matrix, np.ndarray],
              num_iter: int = -1,
              kernel_type: str = "subtree"):
    """Algorithm using Weisefeler-Lehman coloring to check kernels between two graphs.
    Parameters
    ----------
    adjacency_1 : Union[sparse.csr_matrix, np.ndarray]
        First adjacency matrix to be checked.

    adjacency_2 : Union[sparse.csr_matrix, np.ndarray]
        Second adjacency matrix to be checked.

    num_iter : int
    Maximum number of iterations once wants to make. Maximum positive value is the number of nodes in adjacency_1,
    it is also the default value set if given a negative int. We set default value to -1.

    kernel_type : str
        Type of kernel the user wants to check. Default is "subtree". Possible values are "isomorphism", "subtree" and
        "edge".
        Isomorphism only checks if each graph has the same number of each distinct labels at each step.
        Subtree kernel counts the number of occurences of each label in each graph at each step and sums the scalar
        product of these counts.
        Edge kernel counts the number of edge having the same labels for its nodes in both graphs at each step.

    Returns
    -------
    similarity : int
        Likeness between both graphs. -1 if unknown kernel was specified.

    Example
    -------
    >>> from sknetwork.topology import wl_kernel
    >>> from sknetwork.data import house
    >>> adjacency_1 = house()
    >>> adjacency_2 = house()
    >>> similarity = wl_kernel(adjacency_1, adjacency_2, -1, "subtree")
    49

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

    if kernel_type == "isomorphism" :
        return c_wl_kernel(adjacency_1, adjacency_2, num_iter, 1)
    elif kernel_type == "subtree":
        return c_wl_kernel(adjacency_1, adjacency_2, num_iter, 2)
    elif kernel_type ==  "edge":
       return c_wl_kernel(adjacency_1, adjacency_2, num_iter, 3)
    else :
        return -1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int c_wl_kernel(adjacency_1: Union[sparse.csr_matrix, np.ndarray],
                     adjacency_2: Union[sparse.csr_matrix, np.ndarray],
                     int num_iter,
                     int kernel_type):
    """Cythonised function actually running the kernels.
    Parameters
    ----------
    adjacency_1 : Union[sparse.csr_matrix, np.ndarray]
        First adjacency matrix to be checked.

    adjacency_2 : Union[sparse.csr_matrix, np.ndarray]
        Second adjacency matrix to be checked.

    num_iter : int
        Maximum number of iterations once wants to make. Maximum positive value is the number of nodes in adjacency_1,
        it is also the default value set if given a negative int.

    kernel_type : int
        Type of kernel the user wants to check

    Returns
    -------
    similarity : int
        Likeness between both graphs.
    """

    cdef np.ndarray[int, ndim=1] indices_1 = adjacency_1.indices
    cdef np.ndarray[int, ndim=1] indptr_1 = adjacency_1.indptr
    cdef np.ndarray[int, ndim=1] indices_2 = adjacency_2.indices
    cdef np.ndarray[int, ndim=1] indptr_2 = adjacency_2.indptr

    cdef int iteration = 0
    cdef int n = indptr_1.shape[0] - 1
    cdef int m = indptr_2.shape[0] - 1

    if n != m: #TODO in the future had empty nodes
        return 0

    cdef double alpha = - np.pi/3.15
    cdef double [:] powers = np.ones(n, dtype = np.double)

    cdef int k
    for k in range(1,n) :
        powers[k] = powers[k-1]*alpha

    cdef long long[:] labels = np.ones(n, dtype = np.longlong)

    cdef int length_count = 2 * n + 1
    cdef int i
    cdef int similarity = 0
    cdef bint has_changed_1
    cdef bint has_changed_2
    cdef int current_max

    cdef long long[:] labels_1
    cdef long long[:] labels_2

    cdef int[:] count_1
    cdef int[:] count_2

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

    if kernel_type == 1:
        similarity = 1

    while iteration < num_iter : #and (has_changed_1 or has_changed_2), not using this atm cause it gives issues when
                                #not normalizing

        current_max, has_changed_1 = c_wl_coloring(indices_1, indptr_1, 1, labels_1, powers)
        current_max, has_changed_2 = c_wl_coloring(indices_2, indptr_2, 1, labels_2, powers)
        iteration += 1

        if kernel_type == 1:
            if c_wl_isomorphism(labels_1, labels_2, count_1, count_2, n, current_max) == 0:
                return 0
            continue

        if kernel_type == 2:
            similarity = c_wl_subtree_kernel(labels_1, labels_2, count_1, count_2, n, current_max, similarity)
            continue

        if kernel_type == 3:
            similarity = c_wl_edge_kernel(indices_1, indptr_1, indices_2, indptr_2, labels_1, labels_2, n, similarity)
            continue

    return similarity

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int c_wl_isomorphism(long long[:] labels_1,
                          long long[:] labels_2,
                          int[:] count_1,
                          int[:] count_2,
                          int n,
                          int current_max):
    """
    Parameters
    ----------
    labels_1 : long long[:]
        The labels of the first graph.

    labels_2 : long long[:]
        The labels of the second graph.

    count_1 : int[:]
        The counter for the labels of the first graph.

    count_2 : int[:]
         The counter for the labels of the second graph.

    n : int
        Length of labels_1 and labels_2.

    current_max : int
        The maximum label currently in labels_1 and labels_2. Used to save (little) time.

    Returns
    -------
        0 if labels_1 and labels_2 have a different occurrences of one label, 1 otherwise.
    """

    for i in range(current_max + 1):
        count_1[i] = 0
        count_2[i] = 0
    for i in range(n):
        count_1[labels_1[i]] += 1
        count_2[labels_2[i]] += 1
    for i in range(current_max + 1):
        if count_1[i] != count_2[i]:
            return 0

    return 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int c_wl_subtree_kernel(long long[:] labels_1,
                             long long[:] labels_2,
                             int[:] count_1,
                             int[:] count_2,
                             int n,
                             int current_max,
                             int similarity):
    """
    Parameters
    ----------
    labels_1 : long long[:]
        The labels of the first graph.

    labels_2 : long long[:]
        The labels of the second graph.

    count_1 : int[:]
        The counter for the labels of the first graph.

    count_2 : int[:]
         The counter for the labels of the second graph.

    n : int
        Length of labels_1 and labels_2.

    current_max : int
        The maximum label currently in labels_1 and labels_2. Used to save (little) time.

    similarity : int
        The current value of similarity for the two graphs.

    Returns
    -------
        Updated similarity.
    """

    for i in range(current_max + 1):
        count_1[i] = 0
        count_2[i] = 0
    for i in range(n):
        count_1[labels_1[i]] += 1
        count_2[labels_2[i]] += 1
    for i in range(current_max + 1):
        similarity += count_1[i] * count_2[i]

    return similarity

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int c_wl_edge_kernel(np.ndarray[int, ndim=1] indices_1,
                          np.ndarray[int, ndim=1] indptr_1,
                          np.ndarray[int, ndim=1] indices_2,
                          np.ndarray[int, ndim=1] indptr_2,
                          long long[:] labels_1,
                          long long[:] labels_2,
                          int n,
                          int similarity):
    """
    Parameters
    ----------
    indices_1 : np.ndarray[int, ndim=1]
        Indices of the first graph in CSR format.

    indices_2 : np.ndarray[int, ndim=1]
        Indices of the second graph in CSR format.

    indptr_1 : np.ndarray[int, ndim=1]
        Indptr of the first graph in CSR format.

    indptr_2 : np.ndarray[int, ndim=1]
        Indices of the second graph in CSR format.

    labels_1 : long long[:]
        The labels of the first graph.

    labels_2 : long long[:]
        The labels of the second graph.

    n : int
        Length of labels_1 and labels_2.

    similarity : int
        The current value of similarity for the two graphs.

    Returns
    -------
        Updated similarity.
    """

    cdef int v1, v2, n1, n2, j1, j2, jj1, jj2, d1, d2, l1_1, l1_2, l2_1, l2_2

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
