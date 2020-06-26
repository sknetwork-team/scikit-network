# distutils: language = c++
# cython: language_level=3
"""
Created on Jun, 2020
@author: Alexis Barreaux <alexis.barreaux@telecom-paris.fr>
"""

import numpy as np
cimport numpy as np

cdef void counting_sort(int n, int deg, np.int32_t[:] count, long long[:] multiset, long long[:] sorted_multiset)
