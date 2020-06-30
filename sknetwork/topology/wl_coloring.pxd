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

cdef (cmap[long long, long long], int, bint) c_wl_coloring(np.ndarray[int, ndim=1] indices,
                                                            np.ndarray[int, ndim=1] indptr,
                                                            int max_iter,
                                                            long long[:] labels,
                                                            long long[:,:] multiset,
                                                            vector[cpair] large_label,
                                                            int  [:] count,
                                                            int current_max,
                                                            cmap[long long, long long] new_hash,
                                                            bint c_dict)

cdef void c_wl_coloring_2(np.ndarray[int, ndim=1] indices,
                                np.ndarray[int, ndim=1] indptr,
                                int max_iter,
                                   long long[:] labels)

cpdef np.ndarray[long long, ndim=1] wl_coloring(adjacency, int max_iter, np.ndarray[long long, ndim = 1] input_labels )
