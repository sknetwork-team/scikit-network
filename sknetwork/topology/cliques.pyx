# distutils: language = c++
# cython: language_level=3
"""
Created in June 2020
@author: Julien Simonnet <julien.simonnet@etu.upmc.fr>
@author: Yohann Robert <yohann.robert@etu.upmc.fr>
"""
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
from scipy import sparse

cimport cython

from sknetwork.path.dag import get_dag
from sknetwork.topology.core import get_core_decomposition


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class ListingBox:
    cdef int[:] ns
    cdef np.ndarray degrees
    cdef np.ndarray subs
    cdef short[:] lab

    def __cinit__(self, vector[int] indptr, int k):
        cdef int n = indptr.size() - 1
        cdef int i
        cdef int max_deg = 0

        cdef np.ndarray[int, ndim=1] ns = np.empty((k+1,), dtype=np.int32)
        ns[k] = n
        self.ns = ns

        cdef np.ndarray[short, ndim=1] lab = np.full((n,), k, dtype=np.int16)
        self.lab = lab

        cdef np.ndarray[int, ndim=1] deg = np.zeros(n, dtype=np.int32)
        cdef np.ndarray[int, ndim=1] sub = np.zeros(n, dtype=np.int32)

        for i in range(n):
            deg[i] = indptr[i+1] - indptr[i]
            max_deg = max(deg[i], max_deg)
            sub[i] = i

        self.degrees = np.empty((k+1,), dtype=object)
        self.subs = np.empty((k+1,), dtype=object)

        self.degrees[k] = deg
        self.subs[k] = sub

        for i in range(2, k):
            deg = np.zeros(n, dtype=np.int32)
            sub = np.zeros(max_deg, dtype=np.int32)
            self.degrees[i] = deg
            self.subs[i] = sub


@cython.boundscheck(False)
@cython.wraparound(False)
cdef long count_cliques_from_dag(vector[int] indptr, vector[int] indices, int clique_size, ListingBox box):
    cdef int n = indptr.size() - 1
    cdef long n_cliques = 0
    cdef int i, j, k, k_max
    cdef int u, v, w

    if clique_size == 2:
        degree_ = box.degrees[2]
        sub_ = box.subs[2]
        for i in range(box.ns[2]):
            j = sub_[i]
            n_cliques += degree_[j]
        return n_cliques

    sub_ = box.subs[clique_size]
    sub_prev = box.subs[clique_size - 1]
    degree_ = box.degrees[clique_size]
    deg_prev = box.degrees[clique_size - 1]
    for i in range(box.ns[clique_size]):
        u = sub_[i]
        box.ns[clique_size - 1] = 0
        for j in range(indptr[u], indptr[u] + degree_[u]):
            v = indices[j]
            if box.lab[v] == clique_size:
                box.lab[v] = clique_size - 1
                sub_prev[box.ns[clique_size - 1]] = v
                box.ns[clique_size - 1] += 1
                deg_prev[v] = 0
        for j in range(box.ns[clique_size - 1]):
            v = sub_prev[j]
            k = indptr[v]
            k_max = indptr[v] + degree_[v]
            while k < k_max:
                w = indices[k]
                if box.lab[w] == clique_size - 1:
                    deg_prev[v] += 1
                else:
                    k_max -= 1
                    indices[k] = indices[k_max]
                    k -= 1
                    indices[k_max] = w
                k += 1
        n_cliques += count_cliques_from_dag(indptr, indices, clique_size - 1, box)
        for j in range(box.ns[clique_size - 1]):
            v = sub_prev[j]
            box.lab[v] = clique_size
    return n_cliques


def count_cliques(adjacency: sparse.csr_matrix, clique_size: int = 3) -> int:
    """Count the number of cliques of some size.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    clique_size : int
        Clique size (default = 3, corresponding to triangles.

    Returns
    -------
    n_cliques : int
        Number of cliques.

    Example
    -------
    >>> from sknetwork.data import karate_club
    >>> adjacency = karate_club()
    >>> count_cliques(adjacency, 3)
    45

    References
    ----------
    Danisch, M., Balalau, O., & Sozio, M. (2018, April).
    `Listing k-cliques in sparse real-world graphs.
    <https://dl.acm.org/doi/pdf/10.1145/3178876.3186125>`_
    In Proceedings of the 2018 World Wide Web Conference (pp. 589-598).
    """
    if clique_size < 2:
        raise ValueError("The clique size must be at least 2.")

    values = get_core_decomposition(adjacency)
    dag = get_dag(adjacency, order=np.argsort(values))
    indptr = dag.indptr
    indices = dag.indices
    box = ListingBox.__new__(ListingBox, indptr, clique_size)
    n_cliques = count_cliques_from_dag(indptr, indices, clique_size, box)
    return n_cliques
