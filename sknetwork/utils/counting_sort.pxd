# distutils: language = c++
# cython: language_level=3
"""
Created on Jun, 2020
@author: Alexis Barreaux <alexis.barreaux@telecom-paris.fr>
"""

import numpy as np
cimport numpy as np

cdef void counting_sort(int n, int deg, int[:] count, long long[:] multiset, long long[:] sorted_multiset)

cdef void counting_sort_all(int[:] indptr, int[:] indices, long long[:,:] multiset, long long[:] labels)
