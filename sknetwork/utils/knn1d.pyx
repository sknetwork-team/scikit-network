# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
""" One dimensional nearest neighbor search.
Created on Mar, 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""
import numpy as np
cimport numpy as np

from libcpp.vector cimport vector
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def knn1d(float[:] x, int n_neighbors):
    """K nearest neighbors search for 1-dimensional arrays.

    Parameters
    ----------
    x: np.ndarray
        1-d data
    n_neighbors: int
        Number of neighbors to return.
    Returns
    -------
    tuple
        List of source nodes and list of target nodes.
    """
    cdef int n
    cdef int i
    cdef int j
    cdef int ix
    cdef int low
    cdef int hgh
    cdef int neigh
    cdef int val

    cdef vector[int] sorted_ix
    cdef vector[int] row
    cdef vector[int] col
    cdef vector[int] candidates
    cdef vector[int] sorted_candidates
    cdef vector[int] sorted_deltas
    cdef vector[int] tmp

    cdef vector[float] deltas

    n = x.shape[0]
    tmp = np.argsort(x)
    for i in range(n):
        sorted_ix.push_back(tmp[i])

    for i in range(n):
        deltas.clear()
        sorted_candidates.clear()

        ix = sorted_ix[i]
        low = max(0, i - n_neighbors)
        hgh = min(n - 1, i + n_neighbors + 1)
        candidates = sorted_ix[low:hgh]

        for j in range(len(candidates)):
            deltas.push_back(abs(x[candidates[j]] - x[ix]))

        sorted_deltas = np.argsort(deltas)
        for j in range(len(sorted_deltas)):
            val = candidates[sorted_deltas[j]]
            sorted_candidates.push_back(val)
        sorted_candidates = sorted_candidates[:n_neighbors+1]

        for j in range(len(sorted_candidates)):
            neigh = sorted_candidates[j]
            if neigh != ix:
                row.push_back(ix)
                col.push_back(neigh)

    return row, col
