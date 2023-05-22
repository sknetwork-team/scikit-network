# distutils: language = c++
# cython: language_level=3
"""
Created on Apr 2020
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""
cimport cython
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
def diffusion(int[:] indptr, int[:] indices, float[:] data, float[:] scores, float[:] fluid,
              float damping_factor, int n_iter, float tol):
    """One loop of fluid diffusion."""
    cdef int n = fluid.shape[0]
    cdef int i
    cdef int j
    cdef int j1
    cdef int j2
    cdef int jj
    cdef float sent
    cdef float tmp
    cdef float removed
    cdef float restart_prob = 1 - damping_factor
    cdef float residu = restart_prob

    for k in range(n_iter):
        for i in prange(n, nogil=True, schedule='guided'):
            sent = fluid[i]
            if sent > 0:
                scores[i] += sent
                fluid[i] = 0
                j1 = indptr[i]
                j2 = indptr[i+1]
                tmp = sent * damping_factor
                if j2 != j1:
                    for jj in range(j1, j2):
                        j = indices[jj]
                        fluid[j] += tmp * data[jj]
                    removed = sent * restart_prob
                else:
                    removed = sent
                residu -= removed
        if residu < tol * restart_prob:
            return
    return
