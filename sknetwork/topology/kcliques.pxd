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

