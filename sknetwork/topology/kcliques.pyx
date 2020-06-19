# distutils: language = c++
# cython: language_level=3
"""
Created on Jun 3, 2020
@author: Julien Simonnet <julien.simonnet@etu.upmc.fr>
@author: Yohann Robert <yohann.robert@etu.upmc.fr>
"""
import numpy as np
cimport numpy as np
from scipy import sparse

cimport cython

from sknetwork.topology.kcore import CoreDecomposition

ctypedef np.int_t int_type_t
ctypedef np.uint8_t bool_type_t

@cython.boundscheck(False)
@cython.wraparound(False)

#------ Wrapper for integer array -------

cdef class IntArray:

    def __cinit__(self, int n):
        self.arr.reserve(n)

    def __getitem__(self, int key) -> int:
        return self.arr[key]

    def __setitem__(self, int key, int val) -> None:
        self.arr[key] = val

# ----- Collections of arrays used by our listing algorithm -----

cdef class ListingBox:

    def __cinit__(self, int[:] indptr, short k):
        self.initBox(indptr, k)

    # building the special graph structure
    cdef void initBox(self, int[:] indptr, short k):
        cdef int n = indptr.shape[0] - 1
        cdef int i
        cdef int max_deg = 0

        cdef IntArray deg
        cdef IntArray sub

        cdef np.ndarray[int, ndim=1] ns
        cdef np.ndarray[short, ndim=1] lab

        lab = np.full((n,), k, dtype=np.int16)

        deg = IntArray.__new__(IntArray, n)
        sub = IntArray.__new__(IntArray, n)

        for i in range(n):
            deg[i] = indptr[i+1] - indptr[i]
            max_deg = max(deg[i], max_deg)
            sub[i] = i

        self.ns = np.empty((k+1,), dtype=np.int32)
        self.ns[k] = n

        self.degrees = np.empty((k+1,), dtype=object)
        self.subs = np.empty((k+1,), dtype=object)

        self.degrees[k] = deg
        self.subs[k] = sub

        for i in range(2, k):
            deg = IntArray.__new__(IntArray, n)
            sub = IntArray.__new__(IntArray, max_deg)
            self.degrees[i] = deg
            self.subs[i] = sub

        self.lab = lab

# ---------------------------------------------------------------

cdef long listing_rec(int[:] indptr, int[:] indices, short l, ListingBox box):
    cdef int n = indptr.shape[0] - 1
    cdef long nb_cliques
    cdef int i, j, k
    cdef int u, v, w
    cdef int cd
    cdef IntArray sub_l, degre_l

    nb_cliques = 0

    if l == 2:
        degre_l = box.degrees[2]
        sub_l = box.subs[2]
        for i in range(box.ns[2]):
            j = sub_l[i]
            nb_cliques += degre_l[j]

        return nb_cliques

    cdef IntArray deg_prevs, sub_prev
    sub_l = box.subs[l]
    sub_prev = box.subs[l-1]
    degre_l = box.degrees[l]
    deg_prev = box.degrees[l-1]
    for i in range(box.ns[l]):
        u = sub_l[i]
        box.ns[l-1] = 0
        cd = indptr[u] + degre_l[u]
        for j in range(indptr[u], cd):
            v = indices[j]
            if box.lab[v] == l:
                box.lab[v] = l-1
                # box.subs[l-1][ns[l-1]++] = v
                sub_prev[box.ns[l-1]] = v
                box.ns[l-1] += 1
                deg_prev[v] = 0

        for j in range(box.ns[l-1]):
            v = sub_prev[j]
            cd = indptr[v] + degre_l[v]
            k = indptr[v]
            while k < cd:
                w = indices[k]
                if box.lab[w] == l-1:
                    deg_prev[v] += 1
                else:
                    cd -= 1
                    # indices[k--] = indices[--cd]
                    indices[k] = indices[cd]
                    k -= 1
                    indices[cd] = w

                k += 1

        nb_cliques += listing_rec(indptr, indices, l-1, box)
        for j in range(box.ns[l-1]):
            v = sub_prev[j]
            box.lab[v] = l

    return nb_cliques


cdef long fit_core(int n_edges, int[:] indptr, int[:] indices, int[:] cores, short l):

    cdef int n = indptr.shape[0] - 1
    cdef vector[int] indexation
    cdef int i, j, k

    indexation.reserve(n)
    for i in range(n):
        indexation[cores[i]] = i

    cdef int e
    cdef int[:] row, column
    cdef np.ndarray[bool_type_t, ndim=1] data

    row = np.empty((n_edges,), dtype=np.int32)
    column = np.empty((n_edges,), dtype=np.int32)

    e = 0
    for i in range(n):
        for k in range(indptr[i], indptr[i+1]):
            j = indices[k]
            if indexation[i] < indexation[j]:
                row[e] = indexation[i]
                column[e] = indexation[j]
                e += 1

    row = row[:e]
    column = column[:e]
    data = np.ones((e,), dtype=bool)
    dag = sparse.csr_matrix((data, (row, column)), (n, n), dtype=bool)

    return listing_rec(dag.indptr, dag.indices, l, ListingBox.__new__(ListingBox, dag.indptr, l))


class CliqueListing:
    """ Clique listing algorithm which creates a DAG and list all cliques on it.

    * Graphs

    Attributes
    ----------
    nb_cliques : int
        Number of cliques

    Example
    -------
    >>> from sknetwork.topology import CliqueListing
    >>> from sknetwork.data import karate_club
    >>> cl = CliqueListing()
    >>> adjacency = karate_club()
    >>> cl.fit_transform(adjacency, 3)
    45
    """

    def __init__(self):
        self.nb_cliques = 0


    def fit(self, adjacency : sparse.csr_matrix, k : int) -> 'CliqueListing':
        """ k-cliques listing.

        Parameters
        ----------
        adjacency:
            Adjacency matrix of the graph.
        k:
            k value of cliques to list

        Returns
        -------
         self: :class:`CliqueListing`
        """

        if k < 2:
            raise ValueError("k should not be inferior to 2")

        kcore = CoreDecomposition()
        labels = kcore.fit_transform(adjacency)
        sorted_nodes = np.argsort(labels).astype(np.int32)

        self.nb_cliques = fit_core(adjacency.nnz, adjacency.indptr, adjacency.indices, sorted_nodes, k)

        return self

    def fit_transform(self, *args, **kwargs) -> int:
        """ Fit algorithm to the data and return the number of cliques. Same parameters as the ``fit`` method.

        Returns
        -------
        nb_cliques : int
            Number of k-cliques.
        """
        self.fit(*args, **kwargs)
        return self.nb_cliques


