# distutils: language = c++
# cython: language_level=3
"""
Created on Jun 3, 2020
@author: Julien Simonnet <julien.simonnet@etu.upmc.fr>
@author: Yohann Robert <yohann.robert@etu.upmc.fr>
"""

cimport numpy as np

from libcpp.vector cimport vector

cdef class IntArray:
	cdef vector[int] arr

cdef class ListingBox:
	cdef int[:] ns
	cdef IntArray[:] degres
	cdef IntArray[:] subs
	cdef short[:] lab
	
	cdef void initBox(self, int n_nodes, int[:] indptr, short k)

