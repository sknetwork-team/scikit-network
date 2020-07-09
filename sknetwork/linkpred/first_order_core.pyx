# distutils: language = c++
# cython: language_level=3
"""
Created on July, 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""
from libc.math cimport log
from libcpp.vector cimport vector

ctypedef float (*int2float)(int)
ctypedef float (*vectors2float)(vector[int], vector[int])


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


cdef float size_intersection(vector[int] a, vector[int] b):
    """Size of the intersection of two vectors"""
    return vector_intersection(a, b).size()


cdef float jaccard(vector[int] a, vector[int] b):
    """Jaccard coefficient"""
    cdef float size_inter = size_intersection(a, b)
    cdef float size_union = a.size() + b.size() - size_inter
    return size_inter / size_union


cdef vector[int] neighbors(int[:] indptr, int[:] indices, int node):
    """Neighbors of a given node"""
    cdef int j1 = indptr[node]
    cdef int j2 = indptr[node + 1]
    cdef int j
    cdef vector[int] neigh = ()

    for j in range(j1, j2):
        neigh.push_back(indices[j])

    return neigh


cdef vector[float] common_neigh_global(int[:] indptr, int[:] indices, int source, int[:] targets,
                                       vectors2float weight_func):
    """Scores based on global information about common neighbors"""
    cdef int target, i
    cdef int n_targets = targets.shape[0]
    cdef vector[float] preds

    cdef vector[int] neigh_s = neighbors(indptr, indices, source)
    cdef vector[int] neigh_t
    for i in range(n_targets):
        target = targets[i]
        neigh_t = neighbors(indptr, indices, target)
        preds.push_back(weight_func(neigh_s, neigh_t))

    return preds


def n_common_neigh(int[:] indptr, int[:] indices, int source, int[:] targets):
    """Number of common neighbors"""
    return common_neigh_global(indptr, indices, source, targets, size_intersection)


def jaccard_common_neigh(int[:] indptr, int[:] indices, int source, int[:] targets):
    """Jaccard coefficient of common neighbors"""
    return common_neigh_global(indptr, indices, source, targets, jaccard)


cdef vector[float] common_neigh_local(int[:] indptr, int[:] indices, int source, int[:] targets, int2float weight_func):
    """Scores based on local information about common neighbors"""
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
    return common_neigh_local(indptr, indices, source, targets, inv_log)


def resource_allocation(int[:] indptr, int[:] indices, int source, int[:] targets):
    """Resource Allocation index"""
    return common_neigh_local(indptr, indices, source, targets, inv)
