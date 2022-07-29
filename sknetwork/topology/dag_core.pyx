# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
"""
Created on Jun 3, 2020
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""
from libcpp.vector cimport vector

cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def fit_core(int[:] indptr, int[:] indices, int[:] sorted_nodes, int[:] ix):
    """Build DAG given an order of the nodes.
    """
    cdef int n = indptr.shape[0] - 1
    cdef int u, v, k
    cdef long n_triangles = 0
    cdef vector[int] dag_indptr, dag_indices

    for i in range(n):
        ix[sorted_nodes[i]] = i

    # create the DAG
    cdef int ptr = 0
    dag_indptr.push_back(ptr)
    for u in range(n):
        for k in range(indptr[u], indptr[u+1]):
            v = indices[k]
            if ix[u] < ix[v]:	# the edge needs to be added
                dag_indices.push_back(v)
                ptr += 1
        dag_indptr.push_back(ptr)

    return dag_indptr, dag_indices
