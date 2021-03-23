# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
"""
Created on Mars 2021
@author: Wenzhuo Zhao <wenzhuo.zhao@etu.sorbonne-unversite.fr>
"""
import numpy as np
cimport numpy as np
cimport cython
from libcpp.queue cimport queue
from cython.parallel cimport prange


@cython.boundscheck(False)
@cython.wraparound(False)
def push_pagerank(int n, np.ndarray[np.int32_t, ndim=1] degrees, int[:] indptr, int[:] indices, int[:] rev_indptr, int[:] rev_indices, float[:] seeds, float damping_factor, float tol):
    """Push-based PageRank"""
    cdef np.ndarray[np.float32_t, ndim=1] r
    cdef int v
    cdef int w
    cdef int j1
    cdef int j2
    cdef int jj
    cdef int[:] indexes
    cdef int index
    cdef float probability
    cdef queue[int] worklist
    cdef np.ndarray[np.float32_t, ndim=1] scores
    cdef float tmp
    cdef float norm

    r = np.zeros(n, dtype=np.float32)
    for v in prange(n, nogil=True):
        j1 = rev_indptr[v]
        j2 = rev_indptr[v+1]
        for jj in range(j1, j2):
            w = rev_indices[jj]
            r[v] += 1 / degrees[w]
        """add the probability of seeds"""
        r[v] *= (1 - damping_factor) * damping_factor * (1 + seeds[v])

    # node with high residual value will be processed first
    indexes = np.argsort(-r).astype(np.int32)
    for index in indexes:
        worklist.push(index)
    scores = np.full(n, (1-damping_factor), dtype=np.float32)

    while not worklist.empty():
        v = worklist.front()
        worklist.pop()
        # scores[v]_new
        scores[v] += r[v]
        # iterate node v's out-coming neighbors w
        j1 = indptr[v]
        j2 = indptr[v + 1]
        for jj in prange(j1, j2, nogil=True):
            w = indices[jj]
            # r_old[w]
            tmp = r[w]
            r[w] += r[v] * (1 - damping_factor) / degrees[v]
            if r[w] >= tol > tmp:
                worklist.push(w)
    norm = np.linalg.norm(scores, 1)
    scores /= norm
    return scores
