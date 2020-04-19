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

ctypedef np.int_t int_type_t
ctypedef np.float_t float_type_t


@cython.boundscheck(False)
@cython.wraparound(False)
def diffusion(int[:] indptr, int[:] indices, np.float_t[:] data, np.float_t[:] scores, np.float_t[:] fluid,
              float_type_t damping_factor, int_type_t n_iter):
    """One loop of fluid diffusion."""
    cdef int n = len(fluid)
    cdef int i
    cdef int j
    cdef int jj
    cdef float tmp

    for k in range(n_iter):
        with nogil:
            for i in prange(n):
                tmp = fluid[i]
                if tmp > 0:
                    scores[i] += tmp
                    fluid[i] = 0

                    for jj in range(indptr[i], indptr[i+1]):
                        j = indices[jj]
                        fluid[j] += damping_factor * tmp * data[jj]
    return np.array(scores)
