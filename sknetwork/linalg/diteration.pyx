# distutils: language = c++
# cython: language_level=3
"""
Created on Apr 2020
@author: Nathan de Lara <ndelara@enst.fr>
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
    cdef int jj
    cdef float sent
    cdef float tmp
    cdef float residu = 1

    for k in range(n_iter):
        for i in prange(n, nogil=True, schedule='guided'):
            sent = fluid[i]
            if sent > 0:
                scores[i] += sent
                fluid[i] = 0
                tmp = sent * damping_factor
                for jj in range(indptr[i], indptr[i+1]):
                    j = indices[jj]
                    fluid[j] += tmp * data[jj]
                residu -= sent
        if residu < tol:
            return
    return
