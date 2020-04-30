# distutils: language = c++
# cython: language_level=3
"""
Created on April, 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""
from libcpp.set cimport set
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def vote_update(int[:] indptr, int[:] indices, int[:] labels, int[:] index):
    """One pass of label updates over the graph by majority vote among neighbors."""
    cdef int i
    cdef int ii
    cdef int j
    cdef int n_indices = len(index)
    cdef int label
    cdef int best_count

    cdef vector[int] labels_neigh
    cdef set[int] labels_unique = ()
    cdef int[:] counts = np.zeros_like(labels)

    for ii in range(n_indices):
        i = index[ii]
        labels_neigh.clear()
        for j in range(indptr[i], indptr[i + 1]):
            labels_neigh.push_back(labels[indices[j]])

        labels_unique.clear()
        for label in labels_neigh:
            if label >= 0:
                labels_unique.insert(label)
                counts[label] += 1

        best_count = -1
        for label in labels_unique:
            if counts[label] > best_count:
                labels[i] = label
                best_count = counts[label]
            counts[label] = 0
    return np.asarray(labels)
