# distutils: language = c++
# cython: language_level=3
""" Specific counting sort used in topology.wl_coloring.
Created on June, 2020
@author: Alexis Barreaux <alexis.barreaux@telecom-paris.fr>
"""
import numpy as np
cimport numpy as np


cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void counting_sort(int n, int deg, np.int32_t[:] count, np.longlong_t[:] multiset, np.longlong_t[:] sorted_multiset):
    """Sorts an array by using counting sort, variant of bucket sort.

    Parameters
    ----------
    n : int
        The size (number of nodes) of the graph.

    deg: int
        The deg of current node and size of multiset.

    count : np.int32_t[:]
        Buckets to count ocurrences.

    multiset : np.longlong_t[:]
        The array to be sorted.

    sorted_multiset : np.longlong_t[:]
        The array where multiset will be sorted.
    """

    cdef int total = 0
    cdef int i
    cdef int j

    for i in range(n):
        count[i] = 0

    for i in range(deg):
        j =multiset[i]
        count[j] += 1

    for i in range(n):
        j = total
        total+= count[i]
        count[i] = j

    for i in range(deg):
        sorted_multiset[count[multiset[i]]] = multiset[i]
        count[multiset[i]] += 1

    for i in range(deg):
        multiset[i] = sorted_multiset[i]
