# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
"""
Created on Jun 3, 2020
@author: Julien Simonnet <julien.simonnet@etu.upmc.fr>
@author: Yohann Robert <yohann.robert@etu.upmc.fr>
"""
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
from scipy import sparse

cimport cython

from sknetwork.topology.dag import DAG
from sknetwork.topology.kcore import CoreDecomposition


# ----- Collections of arrays used by our listing algorithm -----
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
cdef long fit_core(vector[int] indptr, vector[int] indices, int l, ListingBox box):
    cdef int n = indptr.size() - 1
    cdef long n_cliques = 0
    cdef int i, j, k
    cdef int u, v, w
    cdef int cd

    if l == 2:
        degree_l = box.degrees[2]
        sub_l = box.subs[2]
        for i in range(box.ns[2]):
            j = sub_l[i]
            n_cliques += degree_l[j]

        return n_cliques

    sub_l = box.subs[l]
    sub_prev = box.subs[l-1]
    degree_l = box.degrees[l]
    deg_prev = box.degrees[l-1]
    for i in range(box.ns[l]):
        u = sub_l[i]
        box.ns[l-1] = 0
        cd = indptr[u] + degree_l[u]
        for j in range(indptr[u], cd):
            v = indices[j]
            if box.lab[v] == l:
                box.lab[v] = l-1
                sub_prev[box.ns[l-1]] = v
                box.ns[l-1] += 1
                deg_prev[v] = 0

        for j in range(box.ns[l-1]):
            v = sub_prev[j]
            cd = indptr[v] + degree_l[v]
            k = indptr[v]
            while k < cd:
                w = indices[k]
                if box.lab[w] == l-1:
                    deg_prev[v] += 1
                else:
                    cd -= 1
                    indices[k] = indices[cd]
                    k -= 1
                    indices[cd] = w

                k += 1

        n_cliques += fit_core(indptr, indices, l-1, box)
        for j in range(box.ns[l-1]):
            v = sub_prev[j]
            box.lab[v] = l

    return n_cliques


class Cliques:
    """ Clique counting algorithm.

    Parameters
    ----------
    k : int
        Clique order (e.g., k = 3 means triangles).

    Attributes
    ----------
    n_cliques_ : int
        Number of cliques.

    Example
    -------
    >>> from sknetwork.data import karate_club
    >>> cliques = Cliques(k=3)
    >>> adjacency = karate_club()
    >>> cliques.fit_transform(adjacency)
    45

    References
    ----------
    Danisch, M., Balalau, O., & Sozio, M. (2018, April).
    `Listing k-cliques in sparse real-world graphs.
    <https://dl.acm.org/doi/pdf/10.1145/3178876.3186125>`_
    In Proceedings of the 2018 World Wide Web Conference (pp. 589-598).
    """
    def __init__(self, k: int):
        self.k = np.int32(k)
        self.n_cliques_ = 0

    def fit(self, adjacency: sparse.csr_matrix) -> 'Cliques':
        """K-cliques count.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.

        Returns
        -------
         self: :class:`Cliques`
        """
        if self.k < 2:
            raise ValueError("k should be at least 2")

        kcore = CoreDecomposition()
        labels = kcore.fit_transform(adjacency)
        sorted_nodes = np.argsort(labels)

        dag = DAG()
        dag.fit(adjacency, sorted_nodes)
        indptr = dag.indptr_
        indices = dag.indices_

        box = ListingBox.__new__(ListingBox, indptr, self.k)
        self.n_cliques_ = fit_core(indptr, indices, self.k, box)

        return self

    def fit_transform(self, adjacency: sparse.csr_matrix) -> int:
        """ Fit algorithm to the data and return the number of cliques. Same parameters as the ``fit`` method.

        Returns
        -------
        n_cliques : int
            Number of k-cliques.
        """
        self.fit(adjacency)
        return self.n_cliques_


