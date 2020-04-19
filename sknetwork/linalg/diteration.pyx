#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""
import numpy as np
cimport numpy as np
cimport cython
from libcpp.vector cimport vector

ctypedef np.int_t int_type_t
ctypedef np.float_t float_type_t


@cython.boundscheck(False)
@cython.wraparound(False)
def diffusion(int[:] indptr, int[:] indices, np.float_t[:] data, np.float_t[:] scores, np.float_t[:] fluid,
              float_type_t damping_factor):
    """One loop of fluid diffusion."""
    cdef vector[int] outnodes

    cdef int n = len(fluid)
    cdef int i
    cdef int j
    cdef int jj
    cdef float tmp

    for i in range(n):
        tmp = fluid[i]
        if tmp > 0:
            scores[i] += tmp
            fluid[i] = 0

            for jj in range(indptr[i], indptr[i+1]):
                j = indices[jj]
                fluid[j] += damping_factor * tmp * data[jj]
    return fluid, scores
