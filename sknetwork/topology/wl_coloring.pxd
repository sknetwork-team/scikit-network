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

cdef long long [:] c_wl_coloring(int[:] indices,
                                                        int[:] indptr,
                                                         int max_iter,
                                                         long long[:] labels,
                                                         int max_deg,
                                                         int n,
                                                         cmap[long, long] new_hash,
                                                         int[:] degres,
                                                         long long[:] multiset,
                                                         long long[:] sorted_multiset,
                                                         vector[cpair] large_label,
                                                         int  [:] count,
                                                         int current_max,
                                                         bint clear_dict)

cpdef np.ndarray[long long, ndim=1] wl_coloring(adjacency,int max_iter,np.ndarray[long long, ndim = 1] input_labels )
