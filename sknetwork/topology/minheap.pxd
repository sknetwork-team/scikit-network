# distutils: language = c++
# cython: language_level=3
"""
Created in June 2020
@author: Julien Simonnet <julien.simonnet@etu.upmc.fr>
@author: Yohann Robert <yohann.robert@etu.upmc.fr>
"""
from libcpp.vector cimport vector

cdef class MinHeap:

    cdef vector[int] val, pos
    cdef int size

    cdef int pop_min(self, int[:] scores)
    cdef bint empty(self)
    cdef void swap(self, int x, int y)
    cdef void insert_key(self, int k, int[:] scores)
    cdef void decrease_key(self, int i, int[:] scores)
    cdef void min_heapify(self, int i, int[:] scores)
