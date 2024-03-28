# distutils: language=c++
# cython: language_level=3
"""
Created in June 2020
@author: Julien Simonnet <julien.simonnet@etu.upmc.fr>
@author: Yohann Robert <yohann.robert@etu.upmc.fr>
@author: Nathan de Lara <nathan.delara@polytechnique.org>
@author: Thomas Bonald <bonald@enst.fr>
"""
from libcpp.vector cimport vector
from scipy import sparse
from cython.parallel import prange

from sknetwork.path.dag import get_dag
from sknetwork.utils.check import check_square
from sknetwork.utils.format import directed2undirected
from sknetwork.utils.neighbors import get_degrees

cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef long count_local_triangles_from_dag(int node, vector[int] indptr, vector[int] indices) nogil:
    """Count the number of triangles from a given node in a directed acyclic graph.

    Parameters
    ----------
    node :
        Node.
    indptr :
        CSR format index pointer array of the adjacency matrix of the graph.
    indices :
        CSR format index array of the adjacency matrix of the graph.

    Returns
    -------
    n_triangles :
        Number of triangles.
    """
    cdef int i, j, k
    cdef int neighbor
    cdef long n_triangles = 0

    for k in range(indptr[node], indptr[node + 1]):
        neighbor = indices[k]
        i = indptr[node]
        j = indptr[neighbor]

        while (i < indptr[node + 1]) and (j < indptr[neighbor + 1]):
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
cdef long count_triangles_from_dag(vector[int] indptr, vector[int] indices, bint parallelize):
    """Count the number of triangles in a directed acyclic graph.

    Parameters
    ----------
    indptr :
        CSR format index pointer array of the adjacency matrix of the graph.
    indices :
        CSR format index array of the adjacency matrix of the graph.
    parallelize :
        If ``True``, use a parallel range to count triangles.

    Returns
    -------
    n_triangles :
        Number of triangles in the graph
    """
    cdef int n_nodes = indptr.size() - 1
    cdef int node
    cdef long n_triangles = 0

    if parallelize:
        for node in prange(n_nodes, nogil=True):
            n_triangles += count_local_triangles_from_dag(node, indptr, indices)
    else:
        for node in range(n_nodes):
            n_triangles += count_local_triangles_from_dag(node, indptr, indices)

    return n_triangles

def count_triangles(adjacency: sparse.csr_matrix, parallelize: bool = False) -> int:
    """Count the number of triangles in a graph. The graph is considered undirected.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    parallelize :
        If ``True``, use a parallel range while listing the triangles.

    Returns
    -------
    n_triangles : int
        Number of triangles.

    Example
    -------
    >>> from sknetwork.data import karate_club
    >>> adjacency = karate_club()
    >>> count_triangles(adjacency)
    45
    """
    check_square(adjacency)
    dag = get_dag(directed2undirected(adjacency))
    indptr = dag.indptr
    indices = dag.indices
    n_triangles = count_triangles_from_dag(indptr, indices, parallelize)
    return n_triangles

def get_clustering_coefficient(adjacency: sparse.csr_matrix, parallelize: bool = False) -> float:
    """Get the clustering coefficient of a graph.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    parallelize :
        If ``True``, use a parallel range while listing the triangles.

    Returns
    -------
    coefficient : float
        Clustering coefficient.

    Example
    -------
    >>> from sknetwork.data import karate_club
    >>> adjacency = karate_club()
    >>> np.round(get_clustering_coefficient(adjacency), 2)
    0.26
    """
    n_triangles = count_triangles(adjacency, parallelize)
    degrees = get_degrees(directed2undirected(adjacency))
    degrees = degrees[degrees > 1]
    n_edge_pairs = (degrees * (degrees - 1)).sum() / 2
    coefficient = 3  * n_triangles / n_edge_pairs
    return coefficient
