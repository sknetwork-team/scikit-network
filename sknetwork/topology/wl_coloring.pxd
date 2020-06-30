# distutils: language = c++
# cython: language_level=3
"""
Created on Jun, 2020
@author: Alexis Barreaux <alexis.barreaux@telecom-paris.fr>
"""

import numpy as np
cimport numpy as np


from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map as cmap


ctypedef pair[long long, int] cpair
ctypedef pair[double, int] cpair2

cdef long long [:] c_wl_coloring(np.ndarray[int, ndim=1] indices,
                                np.ndarray[int, ndim=1] indptr,
                                int max_iter,
                                long long[:] labels,
                                cmap[long, long] new_hash,
                                long long[:,:] multiset,
                                vector[cpair] large_label,
                                int  [:] count,
                                bint clear_dict)

cdef long long [:] c_wl_coloring_2(np.ndarray[int, ndim=1] indices,
                                np.ndarray[int, ndim=1] indptr,
                                int max_iter)

cpdef np.ndarray[long long, ndim=1] wl_coloring(adjacency, int max_iter, np.ndarray[long long, ndim = 1] input_labels )
