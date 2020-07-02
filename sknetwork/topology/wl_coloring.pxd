# distutils: language = c++
# cython: language_level=3
"""
Created on Jun, 2020
@author: Alexis Barreaux <alexis.barreaux@telecom-paris.fr>
"""
from typing import Union

import numpy as np
cimport numpy as np
from scipy import sparse

ctypedef (long long, double, int) ctuple

cdef (int, bint) c_wl_coloring(np.ndarray[int, ndim=1] indices,
                        np.ndarray[int, ndim=1] indptr,
                        int max_iter,
                        long long[:] labels,
                        double [:] powers)

cpdef long long[:] wl_coloring(adjacency : Union[sparse.csr_matrix, np.ndarray], int max_iter)
