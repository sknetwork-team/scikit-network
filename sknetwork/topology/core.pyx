# distutils: language = c++
# cython: language_level=3
"""
Created in June 2020
@author: Julien Simonnet <julien.simonnet@etu.upmc.fr>
@author: Yohann Robert <yohann.robert@etu.upmc.fr>
"""
cimport cython

from typing import Union

import numpy as np
cimport numpy as np
from scipy import sparse

from sknetwork.utils.check import check_format
from sknetwork.topology.minheap cimport MinHeap


@cython.boundscheck(False)
@cython.wraparound(False)
cdef compute_core(int[:] indptr, int[:] indices):
    """Compute the core value of each node.

    Parameters
    ----------
    indptr :
        CSR format index array of the adjacency matrix.
    indices :
        CSR format index pointer array of the adjacency matrix.

    Returns
    -------
    labels :
        Core value of each node.
    """
    cdef int n = indptr.shape[0] - 1
    cdef int core_value	= 0	# current/max core value of the graph
    cdef int min_node		# current node of minimum degree
    cdef int i, j, k
    cdef int[:] degrees = np.asarray(indptr)[1:] - np.asarray(indptr)[:n]
    cdef np.ndarray[int, ndim=1] labels = np.empty((n,), dtype=np.int32)
    cdef MinHeap mh = MinHeap.__new__(MinHeap, n)	# minimum heap with an update system

    # insert all nodes in the heap
    for i in range(n):
        mh.insert_key(i, degrees)

    i = n - 1
    while not mh.empty():
        min_node = mh.pop_min(degrees)
        core_value = max(core_value, degrees[min_node])

        # decrease the degree of each neighbor of min_node
        for k in range(indptr[min_node], indptr[min_node+1]):
            j = indices[k]
            degrees[j] -= 1
            mh.decrease_key(j, degrees)		# update the heap to take the new degree into account

        labels[min_node] = core_value
        i -= 1

    return np.asarray(labels)


def get_core_decomposition(adjacency: Union[np.ndarray, sparse.csr_matrix]) -> np.ndarray:
    """Get the k-core decomposition of a graph.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.

    Returns
    -------
    core_values :
         Core value of each node.

    Example
    -------
    >>> from sknetwork.data import karate_club
    >>> adjacency = karate_club()
    >>> core_values = get_core_decomposition(adjacency)
    >>> len(core_values)
    34
    """
    adjacency = check_format(adjacency, allow_empty=True)
    indptr = adjacency.indptr
    indices = adjacency.indices
    return compute_core(indptr, indices)
