# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
"""
Created on July, 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""
from libcpp.vector cimport vector


cdef int size_vector_intersection(vector[int] a, vector[int] b):
    """Number of common elements in two sorted vector. Each element is assumed unique in each vector."""
    cdef int size_intersection = 0
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
            size_intersection += 1
            ix_a += 1
            ix_b += 1

    return size_intersection


cdef vector[int] neighbors(int[:] indptr, int[:] indices, int node):
    """Neighbors of a given node"""
    cdef int j1 = indptr[node]
    cdef int j2 = indptr[node + 1]
    cdef int j
    cdef vector[int] neigh = ()

    for j in range(j1, j2):
        neigh.push_back(indices[j])

    return neigh


def n_common_neigh_edge(int[:] indptr, int[:] indices, int source, int target):
    """Number of common neighbors"""
    cdef vector[int] neigh_s = neighbors(indptr, indices, source)
    cdef vector[int] neigh_t = neighbors(indptr, indices, target)

    return size_vector_intersection(neigh_s, neigh_t)


def n_common_neigh_node(int[:] indptr, int[:] indices, int source):
    """Number of common neighbors with each other node"""
    cdef int target
    cdef int n = indptr.shape[0] - 1
    cdef vector[int] preds

    cdef vector[int] neigh_s = neighbors(indptr, indices, source)
    cdef vector[int] neigh_t
    for target in range(n):
        neigh_t = neighbors(indptr, indices, target)
        preds.push_back(size_vector_intersection(neigh_s, neigh_t))

    return preds

