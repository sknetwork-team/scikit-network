# distutils: language = c++
# cython: language_level=3
"""
Created on Jun 3, 2020
@author: Julien Simonnet <julien.simonnet@etu.upmc.fr>
@author: Yohann Robert <yohann.robert@etu.upmc.fr>
"""
from libcpp.vector cimport vector

cdef class IntArray:
    cdef vector[int] arr

cdef class ListingBox:
    cdef int[:] ns
    cdef IntArray[:] degrees
    cdef IntArray[:] subs
    cdef short[:] lab

    cdef void initBox(self, int[:] indptr, short k)

