# distutils: language = c++
# cython: language_level=3

"""
Created on June 19, 2020
@author: Pierre Pebereau <pierre.pebereau@telecom-paris.fr>

"""

import numpy as np
cimport numpy as np

cimport cython

cpdef insert(BinarySearchTree root, long long key) :
    cinsert(root,key)

cdef void cinsert(BinarySearchTree root, long long key) :
    if root is None :
        root = BinarySearchTree(key)
    else :
        if root.value < key :
            if root.right is None :
                root.right = BinarySearchTree(key)
            else :
                cinsert(root.right, key)
        else :
            if root.left is None:
                root.left = BinarySearchTree(key)
            else:
                cinsert(root.left, key)



cdef class BinarySearchTree:

    cdef BinarySearchTree left
    cdef BinarySearchTree right
    cdef long long value

    def __cinit__(self, long long value):
        self.value = value
        self.left = None
        self.right = None

    cpdef parcours(self, int current,long long[:] arr) :
        self.cparcours(&current, arr)

    cdef void cparcours(self, int * current,long long[:] arr ) :
        if self.left is not None :
            self.left.cparcours(current, arr)
        arr[current[0]] = self.value
        current[0] +=1
        if self.right is not None :
            self.right.cparcours(current, arr)
    cpdef bint search(self, long long value) :
        cdef bint isRoot = value == self.value
        cdef bint isLeft = False
        cdef bint isRight = False

        if not(isRoot) and self.left is not None :
           inLeft = self.left.search(value)

        if not(isRoot) and not(inLeft) and self.right is not None :
            inRight = self.right.search(value)

        return (value == self.value) or inLeft or inRight

    cpdef int size(self) :
        cdef int lsize
        cdef int rsize

        if self.left is not None :
            lsize = self.left.size()
        else :
            lsize = 0

        if self.right is not None :
            rsize = self.right.size()
        else :
            rsize = 0
        return 1 + lsize + rsize

    cpdef long long[:] list(self) :
        cdef int n = self.size()
        cdef long long [:] ret = np.empty(n, dtype = np.longlong)
        cdef int i = 0

        self.cparcours(&i, ret)
        return ret
