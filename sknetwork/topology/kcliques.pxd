# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
"""
Created on Jun 3, 2020
@author: Julien Simonnet <julien.simonnet@etu.upmc.fr>
@author: Yohann Robert <yohann.robert@etu.upmc.fr>
"""
import numpy as np
cimport numpy as np

cdef class ListingBox:
    cdef int[:] ns
    cdef np.ndarray degrees
    cdef np.ndarray subs
    cdef short[:] lab
