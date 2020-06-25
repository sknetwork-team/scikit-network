# distutils: language = c++
# cython: language_level=3
"""
Created on Jun, 2020
@author:
"""

import numpy as np
cimport numpy as np

cdef void counting_sort(int n, int deg, np.int32_t[:] count, np.longlong_t[:] multiset, np.longlong_t[:] sorted_multiset)
