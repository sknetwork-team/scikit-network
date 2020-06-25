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
# This function is only used in sknetwork.topology.wl_coloring.pyx.
# For others uses please wrap it and put it in sknetwork.utils.__init__.py
cdef void counting_sort(int n, int deg, np.int32_t[:] count, np.longlong_t[:] multiset, np.longlong_t[:] sorted_multiset):
    """Sorts an array by using counting sort, variant of bucket sort.

    Parameters
    ----------
    n : int
        The size (number of nodes) of the graph.

    deg: int
        The deg of current node and size of multiset.

    count : np.int32_t[:]
        Buckets to count occurrences.

    multiset : np.longlong_t[:]
        The array to be sorted.

    sorted_multiset : np.longlong_t[:]
        The array where multiset will be sorted.
    """
    cdef int i

    for i in range(n):
        count[i] = 0

    for i in range(deg):
        count[multiset[i]] += 1

    for i in range(1, n):
        count[i] += count[i - 1]

    for i in range(deg):
        sorted_multiset[count[multiset[i]] - 1] = multiset[i]
        count[multiset[i]] -= 1

    for i in range(deg):
        multiset[i] = sorted_multiset[i]
