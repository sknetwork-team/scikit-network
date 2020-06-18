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
from cython.parallel import prange

cimport cython

ctypedef np.int_t int_type_t
ctypedef np.uint8_t bool_type_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef long triangles_c(int[:] indptr, int[:] indices):
    """Count the number of triangles in a DAG.

    Parameters
    ----------
    indices :
        CSR format index array of the normalized adjacency matrix of a DAG.
    indptr :
        CSR format index pointer array of the normalized adjacency matrix of a DAG.

    Returns
    -------
    nb_triangles :
        Number of triangles in the graph
    """
    cdef int n = indptr.shape[0] - 1
    cdef int i, j, k
    cdef int u, v
    cdef long nb_triangles = 0		# number of triangles in the DAG

    for u in range(n):
        for k in range(indptr[u], indptr[u+1]):
            v = indices[k]
            i = indptr[u]
            j = indptr[v]

            # calculate the intersection of neighbors of u and v
            while (i < indptr[u+1]) and (j < indptr[v+1]):
                if indices[i] == indices[j]:
                    i += 1
                    j += 1
                    nb_triangles += 1	# increments the number of triangles
                else :
                    if indices[i] < indices[j]:
                        i += 1
                    else :
                        j += 1

    return nb_triangles


@cython.boundscheck(False)
@cython.wraparound(False)
cdef long tri_intersection(int u, int[:] indptr, int[:] indices) nogil:
    """Counts the number of nodes in the intersection of a node and its neighbors in a DAG.

    Parameters
    ----------
    u :
        Index of the node to study.
    indptr :
        CSR format index pointer array of the normalized adjacency matrix of a DAG.
    indices :
        CSR format index array of the normalized adjacency matrix of a DAG.

    Returns
    -------
    nb_inter :
        Number of nodes in the intersection
    """
    cdef int i, j, k
    cdef int v
    cdef long nb_inter = 0		# number of nodes in the intersection

    for k in range(indptr[u], indptr[u+1]):		# iterates over the neighbors of u
        v = indices[k]
        i = indptr[u]
        j = indptr[v]

        # calculates the intersection of the neighbors of u and v
        while (i < indptr[u+1]) and (j < indptr[v+1]):
            if indices[i] == indices[j]:
                i += 1
                j += 1
                nb_inter += 1	# increments the number of nodes in the intersection
            else :
                if indices[i] < indices[j]:
                    i += 1
                else :
                    j += 1

    return nb_inter

@cython.boundscheck(False)
@cython.wraparound(False)
cdef long triangles_parallel_c(int[:] indptr, int[:] indices):
    """Count the number of triangles in a DAG using a parallel range.

    Parameters
    ----------
    indptr :
        CSR format index pointer array of the normalized adjacency matrix of a DAG.
    indices :
        CSR format index array of the normalized adjacency matrix of a DAG.

    Returns
    -------
    nb_triangles :
        Number of triangles in the graph
    """
    cdef int n = indptr.shape[0] - 1
    cdef int u
    cdef long nb_triangles = 0		# number of triangles

    for u in prange(n, nogil=True):	# parallel range
        nb_triangles += tri_intersection(u, indptr, indices)

    return nb_triangles

@cython.boundscheck(False)
@cython.wraparound(False)
cdef long fit_core(int[:] indptr, int[:] indices, bint parallelize):
    """Counts the number of triangles directly without exporting the graph.

    Parameters
    ----------
    indptr :
        CSR format index pointer array of the normalized adjacency matrix of a DAG.
    indices :
        CSR format index array of the normalized adjacency matrix of a DAG.
    parallelize :
        If ``True`` will use a parallel range to list triangles

    Returns
    -------
    nb_triangles :
        Number of triangles in the graph
    """
    cdef int n = indptr.shape[0] - 1
    cdef int[:] degrees, sorted_nodes, ix
    cdef int u, v, k
    cdef vector[int] row, col, data

    degrees = np.asarray(indptr[1:]) - np.asarray(indptr[:-1])
    sorted_nodes = np.argsort(degrees).astype(np.int32)
    ix = np.empty((n,), dtype=np.int32)	# initializes an empty array
    for i in range(n):
        ix[sorted_nodes[i]] = i

    for u in range(n):
        for k in range(indptr[u], indptr[u+1]):
            v = indices[k]
            if ix[u] < ix[v]:	# the edge needs to be added
                row.push_back(ix[u])
                col.push_back(ix[v])
                data.push_back(1)

    dag = sparse.csr_matrix((data, (row, col)), (n, n), dtype=bool)

    # counts/list the triangles in the DAG
    if parallelize:
        return triangles_parallel_c(dag.indptr, dag.indices)
    else:
        return triangles_c(dag.indptr, dag.indices)


class TriangleListing:
    """Triangle listing algorithm which creates a DAG and list all triangles on it.

    * Graphs

    Parameters
    ----------
    parallelize :
        If ``True``, uses a parallel range while listing the triangles.

    Attributes
    ----------
    nb_tri : int
        Number of triangles

    Example
    -------
    >>> from sknetwork.topology import TriangleListing
    >>> from sknetwork.data import karate_club
    >>> tri = TriangleListing()
    >>> adjacency = karate_club()
    >>> tri.fit_transform(adjacency)
    45
    """
    def __init__(self, parallelize : bool = False):
        self.parallelize = parallelize
        self.nb_tri = None

    def fit(self, adjacency : sparse.csr_matrix) -> 'TriangleListing':
        """Count triangles.

        Parameters
        ----------
        adjacency:
            Adjacency matrix of the graph.

        Returns
        -------
         self: :class:`TriangleListing`
        """
        indptr = adjacency.indptr.astype(np.int32)
        indices = adjacency.indices.astype(np.int32)
        self.nb_tri = fit_core(indptr, indices, self.parallelize)

        return self

    def fit_transform(self, *args, **kwargs) -> int:
        """ Fit algorithm to the data and return the number of triangles. Same parameters as the ``fit`` method.

        Returns
        -------
        nb_tri : int
            Number of triangles.
        """
        self.fit(*args, **kwargs)
        return self.nb_tri
