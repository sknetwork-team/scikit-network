# distutils: language = c++
# cython: language_level=3
"""
Created on July, 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""
from libc.math cimport log
from libcpp.vector cimport vector

ctypedef float (*f_type)(int)


cdef float inv(int a):
    """Inverse function"""
    return 1 / a


cdef float inv_log(int a):
    """Inverse of log function"""
    return 1 / log(a)


cdef vector[int] vector_intersection(vector[int] a, vector[int] b):
    """Common elements in two sorted vectors. Each element is assumed unique in each vector."""
    cdef vector[int] intersection
    cdef int e_a, e_b
    cdef int ix_a = 0
    cdef int ix_b = 0
    cdef int size_a = a.size()
    cdef int size_b = b.size()

    while ix_a < size_a and ix_b < size_b:
        e_a = a[ix_a]
        e_b = b[ix_b]

        if e_a < e_b:
            ix_a += 1
        elif e_b < e_a:
            ix_b += 1
        else:
            intersection.push_back(e_a)
            ix_a += 1
            ix_b += 1

    return intersection


cdef vector[int] neighbors(int[:] indptr, int[:] indices, int node):
    """Neighbors of a given node"""
    cdef int j1 = indptr[node]
    cdef int j2 = indptr[node + 1]
    cdef int j
    cdef vector[int] neigh = ()

    for j in range(j1, j2):
        neigh.push_back(indices[j])

    return neigh


def n_common_neigh(int[:] indptr, int[:] indices, int source, int[:] targets):
    """Number of common neighbors with each other node"""
    cdef int target, i
    cdef int n_targets = targets.shape[0]
    cdef vector[int] preds

    cdef vector[int] neigh_s = neighbors(indptr, indices, source)
    cdef vector[int] neigh_t
    for i in range(n_targets):
        target = targets[i]
        neigh_t = neighbors(indptr, indices, target)
        preds.push_back(vector_intersection(neigh_s, neigh_t).size())

    return preds


cdef weighted_common_neigh(int[:] indptr, int[:] indices, int source, int[:] targets, f_type weight_func):
    """Generic function that assign a weight to each common neighbor"""
    cdef int target, i, j
    cdef int n_targets = targets.shape[0]
    cdef float weight
    cdef vector[int] intersection
    cdef vector[float] preds

    cdef vector[int] neigh_s = neighbors(indptr, indices, source)
    cdef vector[int] neigh_t
    for i in range(n_targets):
        target = targets[i]
        neigh_t = neighbors(indptr, indices, target)
        intersection = vector_intersection(neigh_s, neigh_t)

        weight = 0
        for j in intersection:
            weight += weight_func(indptr[j+1] - indptr[j])
        preds.push_back(weight)

    return preds


def adamic_adar(int[:] indptr, int[:] indices, int source, int[:] targets):
    """Adamic Adar index"""
    return weighted_common_neigh(indptr, indices, source, targets, inv_log)


def resource_allocation(int[:] indptr, int[:] indices, int source, int[:] targets):
    """Resource Allocation index"""
    return weighted_common_neigh(indptr, indices, source, targets, inv)
