# distutils: language = c++
# cython: language_level=3
"""
Created on Jun 3, 2020
@author: Julien Simonnet <julien.simonnet@etu.upmc.fr>
@author: Yohann Robert <yohann.robert@etu.upmc.fr>
"""
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
from scipy import sparse
from scipy.special import comb
from cython.parallel import prange

from sknetwork.utils.base import Algorithm

cimport cython

ctypedef np.int_t int_type_t
ctypedef np.uint8_t bool_type_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef long count_local_triangles(int source, int[:] indptr, int[:] indices) nogil:
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
cdef long fit_core(int[:] indptr, int[:] indices, int[:] sorted_nodes, bint parallelize):
    """Counts the number of triangles directly without exporting the graph.

    Parameters
    ----------
    indptr :
        CSR format index pointer array of the normalized adjacency matrix of a DAG.
    indices :
        CSR format index array of the normalized adjacency matrix of a DAG.
    sorted_nodes :
        Index of the nodes, sorted by increasing degree.
    parallelize :
        If ``True``, use a parallel range to count triangles.

    Returns
    -------
    n_triangles :
        Number of triangles in the graph
    """
    cdef int n = indptr.shape[0] - 1
    cdef int[:] ix
    cdef int u, v, k
    cdef long n_triangles = 0
    cdef vector[int] row, col, data

    ix = np.empty((n,), dtype=np.int32)	# initializes an empty array
    for i in range(n):
        ix[sorted_nodes[i]] = i

    # create the DAG
    for u in range(n):
        for k in range(indptr[u], indptr[u+1]):
            v = indices[k]
            if ix[u] < ix[v]:	# the edge needs to be added
                row.push_back(ix[u])
                col.push_back(ix[v])
                data.push_back(1)

    dag = sparse.csr_matrix((data, (row, col)), (n, n), dtype=bool)
    cdef int[:] dag_indptr = dag.indptr
    cdef int[:] dag_indices = dag.indices

    if parallelize:
        for u in prange(n, nogil=True):
            n_triangles += count_local_triangles(u, dag_indptr, dag_indices)
    else:
        for u in range(n):
            n_triangles += count_local_triangles(u, dag_indptr, dag_indices)

    return n_triangles


class Triangles(Algorithm):
    """Triangle count which creates a DAG and counts all triangles on it.

    * Graphs

    Parameters
    ----------
    parallelize :
        If ``True``, uses a parallel range while listing the triangles.

    Attributes
    ----------
    n_triangles_ : int
        Number of triangles
    clustering_coeff_ : float
        Global clustering coefficient of the graph

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
        self.clustering_coeff_ = None

    def fit(self, adjacency : sparse.csr_matrix) -> 'Triangles':
        """Count triangles.

        Parameters
        ----------
        adjacency:
            Adjacency matrix of the graph.

        Returns
        -------
         self: :class:`Triangles`
        """
        indptr = adjacency.indptr.astype(np.int32)
        indices = adjacency.indices.astype(np.int32)

        degrees = indptr[1:] - indptr[:-1]
        sorted_nodes = np.argsort(degrees).astype(np.int32)
        edge_pairs = comb(degrees, 2).sum()

        self.n_triangles_ = fit_core(indptr, indices, sorted_nodes, self.parallelize)
        if edge_pairs > 0:
            self.clustering_coeff_ = 3 * self.n_triangles_ / edge_pairs
        else:
            self.clustering_coeff_ = 0.

        return self

    def fit_transform(self, *args, **kwargs) -> int:
        """ Fit algorithm to the data and return the number of triangles. Same parameters as the ``fit`` method.

        Returns
        -------
        n_triangles_ : int
            Number of triangles.
        """
        self.fit(*args, **kwargs)
        return self.n_triangles_
