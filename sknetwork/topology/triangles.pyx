# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
"""
Created on Jun 3, 2020
@author: Julien Simonnet <julien.simonnet@etu.upmc.fr>
@author: Yohann Robert <yohann.robert@etu.upmc.fr>
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""
from libcpp.vector cimport vector
from scipy import sparse
from scipy.special import comb
from cython.parallel import prange

from sknetwork.topology.dag import DAG
from sknetwork.utils.base import Algorithm

cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef long count_local_triangles(int source, vector[int] indptr, vector[int] indices) nogil:
    """Counts the number of nodes in the intersection of a node and its neighbors in a DAG.

    Parameters
    ----------
    source :
        Index of the node to study.
    indptr :
        CSR format index pointer array of the normalized adjacency matrix of a DAG.
    indices :
        CSR format index array of the normalized adjacency matrix of a DAG.

    Returns
    -------
    n_triangles :
        Number of nodes in the intersection
    """
    cdef int i, j, k
    cdef int v
    cdef long n_triangles = 0		# number of nodes in the intersection

    for k in range(indptr[source], indptr[source+1]):
        v = indices[k]
        i = indptr[source]
        j = indptr[v]

        # calculates the intersection of the neighbors of u and v
        while (i < indptr[source+1]) and (j < indptr[v+1]):
            if indices[i] == indices[j]:
                i += 1
                j += 1
                n_triangles += 1
            else :
                if indices[i] < indices[j]:
                    i += 1
                else :
                    j += 1

    return n_triangles


@cython.boundscheck(False)
@cython.wraparound(False)
cdef long fit_core(vector[int] indptr, vector[int] indices, bint parallelize):
    """Counts the number of triangles directly without exporting the graph.

    Parameters
    ----------
    indptr :
        CSR format index pointer array of the normalized adjacency matrix of a DAG.
    indices :
        CSR format index array of the normalized adjacency matrix of a DAG.
    parallelize :
        If ``True``, use a parallel range to count triangles.

    Returns
    -------
    n_triangles :
        Number of triangles in the graph
    """
    cdef int n = indptr.size() - 1
    cdef int u
    cdef long n_triangles = 0

    if parallelize:
        for u in prange(n, nogil=True):
            n_triangles += count_local_triangles(u, indptr, indices)
    else:
        for u in range(n):
            n_triangles += count_local_triangles(u, indptr, indices)

    return n_triangles


class Triangles(Algorithm):
    """Count the number of triangles in a graph, and evaluate the clustering coefficient.

    * Graphs

    Parameters
    ----------
    parallelize :
        If ``True``, use a parallel range while listing the triangles.

    Attributes
    ----------
    n_triangles_ : int
        Number of triangles.
    clustering_coef_ : float
        Global clustering coefficient of the graph.

    Example
    -------
    >>> from sknetwork.data import karate_club
    >>> triangles = Triangles()
    >>> adjacency = karate_club()
    >>> triangles.fit_transform(adjacency)
    45
    """
    def __init__(self, parallelize : bool = False):
        super(Triangles, self).__init__()
        self.parallelize = parallelize
        self.n_triangles_ = None
        self.clustering_coef_ = None

    def fit(self, adjacency: sparse.csr_matrix) -> 'Triangles':
        """Count triangles.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.

        Returns
        -------
         self: :class:`Triangles`
        """
        degrees = adjacency.indptr[1:] - adjacency.indptr[:-1]
        edge_pairs = comb(degrees, 2).sum()

        dag = DAG(ordering='degree')
        dag.fit(adjacency)
        indptr = dag.indptr_
        indices = dag.indices_

        self.n_triangles_ = fit_core(indptr, indices, self.parallelize)
        if edge_pairs > 0:
            self.clustering_coef_ = 3 * self.n_triangles_ / edge_pairs
        else:
            self.clustering_coef_ = 0.

        return self

    def fit_transform(self, adjacency: sparse.csr_matrix) -> int:
        """ Fit algorithm to the data and return the number of triangles. Same parameters as the ``fit`` method.

        Returns
        -------
        n_triangles_ : int
            Number of triangles.
        """
        self.fit(adjacency)
        return self.n_triangles_
