# distutils: language = c++
# cython: language_level=3
"""
Created in April 2020
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""
from libcpp.set cimport set
from libcpp.vector cimport vector

cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def vote_update(int[:] indptr, int[:] indices, float[:] data, int[:] labels, int[:] index):
    """One pass of label updates over the graph by majority vote among neighbors."""
    cdef int i
    cdef int ii
    cdef int j
    cdef int jj
    cdef int n_indices = index.shape[0]
    cdef int label
    cdef int label_neigh_size
    cdef float best_score

    cdef vector[int] labels_neigh
    cdef vector[float] votes_neigh, votes
    cdef set[int] labels_unique = ()

    cdef int n = labels.shape[0]
    for i in range(n):
        votes.push_back(0)

    for ii in range(n_indices):
        i = index[ii]
        labels_neigh.clear()
        for j in range(indptr[i], indptr[i + 1]):
            jj = indices[j]
            labels_neigh.push_back(labels[jj])
            votes_neigh.push_back(data[jj])

        labels_unique.clear()
        label_neigh_size = labels_neigh.size()
        for jj in range(label_neigh_size):
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
    return labels
