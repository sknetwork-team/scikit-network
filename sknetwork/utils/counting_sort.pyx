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
cdef void counting_sort(int length_count, int deg, int [:] count, long long[:] multiset, long long[:] sorted_multiset):
    """Sorts an array by using counting sort, variant of bucket sort.

    Parameters
    ----------
    length_count : int
        The size of count.

    deg: int
        The deg of current node and size of multiset.

    count : np.int32_t[:]
        Buckets to count occurrences.

    multiset : long long[:]
        The array to be sorted.

    sorted_multiset : long long[:]
        The array where multiset will be sorted.
    """

    cdef int total = 0
    cdef int i
    cdef int j

    for i in range(length_count):
        count[i] = 0

    for i in range(deg):
        j =multiset[i]
        count[j] += 1

    for i in range(length_count):
        j = total
        total+= count[i]
        count[i] = j

    for i in range(deg):
        sorted_multiset[count[multiset[i]]] = multiset[i]
        count[multiset[i]] += 1

    for i in range(deg):
        multiset[i] = sorted_multiset[i]
