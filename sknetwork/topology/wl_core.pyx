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

from libcpp.algorithm cimport sort as csort
from libcpp.vector cimport vector
from libc.math cimport pow as cpowl
cimport cython

ctypedef (long long, double, int) ctuple


cdef bint is_lower(ctuple a, ctuple b) :
    """Lexicographic comparison between triplets based on the first two values.

    Parameters
    ----------
    a:
        First triplet.
    b:
        Second triplet.

    Returns
    -------
    ``True`` if a < b, and ``False`` otherwise.
    """
    cdef long long a1, b1
    cdef double a2, b2
    cdef int a3, b3

    a1, a2, a3 = a
    b1, b2, b3 = b
    if a1 == b1 :
        return a2 < b2
    return a1 < b1


@cython.boundscheck(False)
@cython.wraparound(False)
def c_wl_coloring(np.ndarray[int, ndim=1] indices, np.ndarray[int, ndim=1] indptr, int max_iter, long long[:] labels,
                  double [:] powers):
    """Weisfeiler-Lehman coloring.

    Parameters
    ----------
    indices : np.ndarray[int, ndim=1]
        Indices of the graph in CSR format.
    indptr : np.ndarray[int, ndim=1]
        Indptr of the second graph in CSR format.
    max_iter : int
        Maximum number of iterations once wants to make.
    labels : long long[:]
        Labels to be changed.
    powers : double [:]
        Powers being used as hash and put in a memory view to limit several identical calculations.

    Returns
    -------
    current_max : int
        Used in wl_kernel to limit a loop.
    has_changed : bint
        Used in wl_kernel to limit a loop.
    """
    cdef int iteration, i, j, j1, j2, jj, u, current_max
    cdef double temp_pow
    cdef int n = indptr.shape[0] -1

    cdef double epsilon = cpowl(10, -10)

    cdef vector[ctuple] new_labels
    cdef double current_hash
    cdef ctuple current_tuple
    cdef ctuple previous_tuple

    cdef long long current_label, previous_label
    cdef double previous_hash
    cdef int current_vertex, previous_vertex
    cdef bint has_changed

    cdef float alpha = powers[1]
    for i in range(2, n) :
        powers[i] = powers[i-1] * alpha

    iteration = 0
    has_changed = True

    while iteration <= max_iter and has_changed:
        new_labels.clear()
        for i in range(n):# going through the neighbors of v.
            current_hash = 0
            j1 = indptr[i]
            j2 = indptr[i + 1]
            for jj in range(j1,j2):
                u = indices[jj]

                current_hash += powers[labels[u]]
            new_labels.push_back((labels[i], current_hash, i))
        csort(new_labels.begin(), new_labels.end(),is_lower)

        current_max = 0
        current_tuple = new_labels[0]
        current_label, current_hash, current_vertex = current_tuple
        labels[current_vertex] = current_max
        has_changed = False
        for jj in range(1,n):
            previous_tuple = current_tuple
            current_tuple = new_labels[jj]
            previous_label, previous_hash, previous_vertex = previous_tuple
            current_label, current_hash, current_vertex = current_tuple
            if abs(previous_hash - current_hash) > epsilon or previous_label != current_label :
                current_max+=1

            if not has_changed:
                has_changed = labels[current_vertex] != current_max

            labels[current_vertex] = current_max

        iteration+=1

    return labels, current_max, has_changed

@cython.boundscheck(False)
@cython.wraparound(False)
def c_wl_kernel(adjacency_1: Union[sparse.csr_matrix, np.ndarray], adjacency_2: Union[sparse.csr_matrix, np.ndarray],
                int kernel, int n_iter) -> int:
    """Core kernel function.

    Parameters
    ----------
    adjacency_1 : Union[sparse.csr_matrix, np.ndarray]
        First adjacency matrix to be checked.
    adjacency_2 : Union[sparse.csr_matrix, np.ndarray]
        Second adjacency matrix to be checked.
    kernel : int
        Type of kernel to use.
    n_iter : int
        Maximum number of iterations once wants to make. Maximum positive value is the number of nodes in adjacency_1,
        it is also the default value set if given a negative int.

    Returns
    -------
    similarity : int
        Similarity score between both graphs.
    """
    cdef np.ndarray[int, ndim=1] indices_1 = adjacency_1.indices
    cdef np.ndarray[int, ndim=1] indptr_1 = adjacency_1.indptr
    cdef np.ndarray[int, ndim=1] indices_2 = adjacency_2.indices
    cdef np.ndarray[int, ndim=1] indptr_2 = adjacency_2.indptr

    cdef int iteration = 0
    cdef int last_update
    cdef int temp
    cdef int n = indptr_1.shape[0] - 1
    cdef int m = indptr_2.shape[0] - 1

    if n != m: #TODO in the future add empty nodes
        return 0

    cdef double alpha = - np.pi / 3.15
    cdef double [:] powers = np.ones(n, dtype=np.double)

    cdef int k
    for k in range(1, n) :
        powers[k] = powers[k-1] * alpha

    cdef long long[:] labels = np.ones(n, dtype=np.longlong)

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

    count_1 = np.zeros(length_count, dtype=np.int32)
    count_2 = np.zeros(length_count, dtype=np.int32)
    labels_1 = np.ones(n, dtype=np.longlong)
    labels_2 = np.ones(n, dtype=np.longlong)
    large_label = np.zeros((n, 2), dtype=np.int32)

    if n_iter > 0 :
        n_iter = min(n, n_iter)
    else :
        n_iter = n

    if kernel == 1:
        similarity = 1

    while iteration < n_iter and (has_changed_1 or has_changed_2):

        _, current_max, has_changed_1 = c_wl_coloring(indices_1, indptr_1, 1, labels_1, powers)
        _, current_max, has_changed_2 = c_wl_coloring(indices_2, indptr_2, 1, labels_2, powers)
        iteration += 1

        if kernel == 1:
            if c_wl_isomorphism(labels_1, labels_2, count_1, count_2, n, current_max) == 0:
                return 0
            continue

        if kernel == 2:
            temp = c_wl_subtree_kernel(labels_1, labels_2, count_1, count_2, n, current_max, similarity)
            last_update = temp - similarity
            similarity = temp
            continue

        if kernel == 3:
            temp = c_wl_edge_kernel(indices_1, indptr_1, indices_2, indptr_2, labels_1, labels_2, n, similarity)
            last_update = temp - similarity
            similarity = temp
            continue

    #If we test isomorphism we only have to give back similarity
    if kernel == 1:
        return similarity

    #otherwise we might have to update similarity because we stopped early to save time.
    similarity += last_update * (n_iter - iteration)

    return similarity

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int c_wl_isomorphism(long long[:] labels_1, long long[:] labels_2, int[:] count_1, int[:] count_2, int n,
                          int current_max):
    """Isomorphism kernel. 0 if labels_1 and labels_2 have a different occurrences of one label, 1 otherwise.

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
    similarity : int
    """
    cdef int i

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
cdef int c_wl_subtree_kernel(long long[:] labels_1, long long[:] labels_2, int[:] count_1, int[:] count_2, int n,
                             int current_max, int similarity):
    """Subtree kernel

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
        The maximum label in labels_1 and labels_2.
    similarity : int
        The current value of similarity for the two graphs.

    Returns
    -------
    similarity : int
        Updated similarity.
    """
    cdef int i

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
cdef int c_wl_edge_kernel(np.ndarray[int, ndim=1] indices_1, np.ndarray[int, ndim=1] indptr_1,
                          np.ndarray[int, ndim=1] indices_2, np.ndarray[int, ndim=1] indptr_2, long long[:] labels_1,
                          long long[:] labels_2, int n, int similarity):
    """Edge kernel

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
    similarity: int
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

                            if (l1_1==l2_1 and l1_2==l2_2) or (l1_2==l2_1 and l1_1==l2_2): #compare ordered pairs
                                similarity+=1
    return similarity
