# distutils: language = c++
# cython: language_level=3
"""
Created on Jun 3, 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
from scipy import sparse

from sknetwork.utils.base import Algorithm

cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef fit_core(int[:] indptr, int[:] indices, int[:] sorted_nodes):
    """Build DAG given an order of the nodes.
    """
    cdef int n = indptr.shape[0] - 1
    cdef int[:] ix
    cdef int u, v, k
    cdef long n_triangles = 0
    cdef vector[int] dag_indptr, dag_indices

    ix = np.empty((n,), dtype=np.int32)	# initializes an empty array
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


class DAG(Algorithm):
    """Build a Directed Acyclic Graph from an adjacency.
    """
    def __init__(self, ordering: str = None):
        super(DAG, self).__init__()
        self.ordering = ordering
        self.indptr_ = None
        self.indices_ = None

    def fit(self, adjacency: sparse.csr_matrix):
        """Fit algorithm to the data."""
        indptr = adjacency.indptr.astype(np.int32)
        indices = adjacency.indices.astype(np.int32)

        if self.ordering is None:
            sorted_nodes = np.arange(adjacency.shape[0]).astype(np.int32)
        elif self.ordering == 'degree':
            degrees = indptr[1:] - indptr[:-1]
            sorted_nodes = np.argsort(degrees).astype(np.int32)
        else:
            raise ValueError('Unknown ordering of nodes.')

        dag_indptr, dag_indices = fit_core(indptr, indices, sorted_nodes)
        self.indptr_ = dag_indptr
        self.indices_ = dag_indices

        return self
