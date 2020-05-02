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
def vote_update(int[:] indptr, int[:] indices, float[:] data, int[:] labels, int[:] index):
    """One pass of label updates over the graph by majority vote among neighbors."""
    cdef int i
    cdef int ii
    cdef int j
    cdef int jj
    cdef int n_indices = len(index)
    cdef int label
    cdef float best_score

    cdef vector[int] labels_neigh
    cdef vector[float] votes_neigh
    cdef set[int] labels_unique = ()
    cdef float[:] votes = np.zeros_like(labels, dtype=np.float32)

    for ii in range(n_indices):
        i = index[ii]
        labels_neigh.clear()
        for j in range(indptr[i], indptr[i + 1]):
            jj = indices[j]
            labels_neigh.push_back(labels[jj])
            votes_neigh.push_back(data[jj])

        labels_unique.clear()
        for jj in range(labels_neigh.size()):
            label = labels_neigh[jj]
            if label >= 0:
                labels_unique.insert(label)
                votes[label] += votes_neigh[jj]

        best_score = -1
        for label in labels_unique:
            if votes[label] > best_score:
                labels[i] = label
                best_score = votes[label]
            votes[label] = 0
    return np.asarray(labels)
