# distutils: language = c++
# cython: language_level=3
"""
Created on Apr 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
def diffusion(int[:] indptr, int[:] indices, float[:] data, float[:] scores, float[:] fluid,
              float damping_factor, int n_iter):
    """One loop of fluid diffusion."""
    cdef int n = len(fluid)
    cdef int i
    cdef int j
    cdef int jj
    cdef float tmp1
    cdef float tmp2

    for k in range(n_iter):
        for i in prange(n, nogil=True):
            tmp1 = fluid[i]
            if tmp1 > 0:
                scores[i] += tmp1
                fluid[i] = 0
                tmp2 = tmp1 * damping_factor
                for jj in range(indptr[i], indptr[i+1]):
                    j = indices[jj]
                    fluid[j] += tmp2 * data[jj]
    return np.asarray(scores)
