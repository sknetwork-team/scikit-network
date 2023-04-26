# distutils: language = c++
# cython: language_level=3
"""
Created on Mars 2021
@author: Wenzhuo Zhao <wenzhuo.zhao@etu.sorbonne-universite.fr>
"""
from libcpp.queue cimport queue
from cython.parallel cimport prange
import numpy as np
cimport numpy as cnp
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def push_pagerank(int n, cnp.ndarray[cnp.int32_t, ndim=1] degrees,
                  int[:] indptr, int[:] indices,
                  int[:] rev_indptr, int[:] rev_indices,
                  cnp.ndarray[cnp.float32_t, ndim=1] seeds,
                  cnp.float32_t damping_factor, cnp.float32_t tol):
    """Push-based PageRank"""
    cdef cnp.ndarray[cnp.float32_t, ndim=1] residuals
    cdef int vertex
    cdef int neighbor
    cdef int j1
    cdef int j2
    cdef int j
    cdef int[:] indexes
    cdef int index
    cdef float probability
    cdef queue[int] worklist
    cdef cnp.ndarray[cnp.float32_t, ndim=1] scores
    cdef cnp.float32_t tmp
    cdef float norm

    residuals = np.zeros(n, dtype=np.float32)
    for vertex in prange(n, nogil=True):
        j1 = rev_indptr[vertex]
        j2 = rev_indptr[vertex + 1]
        # iterate node's in-coming neighbors
        for j in range(j1, j2):
            neighbor = rev_indices[j]
            residuals[vertex] += 1 / degrees[neighbor]
        """add the probability of seeds"""
        residuals[vertex] *= (1 - damping_factor) * \
            damping_factor * (1 + seeds[vertex])

    # node with high residual value will be processed first
    indexes = np.argsort(-residuals).astype(np.int32)
    for index in indexes:
        worklist.push(index)
    scores = np.full(n, (1 - damping_factor), dtype=np.float32)

    while not worklist.empty():
        vertex = worklist.front()
        worklist.pop()
        # scores[v]_new
        scores[vertex] += residuals[vertex]
        # iterate node's out-coming neighbors
        j1 = indptr[vertex]
        j2 = indptr[vertex + 1]
        for j in prange(j1, j2, nogil=True):
            neighbor = indices[j]
            tmp = residuals[neighbor]
            residuals[neighbor] += residuals[vertex] * \
                (1 - damping_factor) / degrees[vertex]
            if residuals[neighbor] > tol > tmp:
                worklist.push(neighbor)
    norm = np.linalg.norm(scores, 1)
    scores /= norm
    return scores
