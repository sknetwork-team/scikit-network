# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
"""
Created on Jun 3, 2020
@author: Julien Simonnet <julien.simonnet@etu.upmc.fr>
@author: Yohann Robert <yohann.robert@etu.upmc.fr>
"""
cimport cython

import numpy as np
cimport numpy as np

from scipy import sparse

from sknetwork.utils.base import Algorithm
from sknetwork.utils.minheap cimport MinHeap


@cython.boundscheck(False)
@cython.wraparound(False)
cdef fit_core(int[:] indptr, int[:] indices):
    """Compute the core value of each node.

    Parameters
    ----------
    indptr :
        CSR format index array of the normalized adjacency matrix.
    indices :
        CSR format index pointer array of the normalized adjacency matrix.

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

    # inserts all nodes in the heap
    for i in range(n):
        mh.insert_key(i, degrees)

    i = n - 1		# index of the rear of the list/array
    while not mh.empty():		# until the heap is emptied
        min_node = mh.pop_min(degrees)
        core_value = max(core_value, degrees[min_node])

        # decreases the degree of each neighbors of min_node to simulate its deletion
        for k in range(indptr[min_node], indptr[min_node+1]):
            j = indices[k]
            degrees[j] -= 1
            mh.decrease_key(j, degrees)		# updates the heap to take into account the new degrees

        labels[min_node] = core_value
        i -= 1

    return np.asarray(labels)


class CoreDecomposition(Algorithm):
    """K-core decomposition algorithm.

    * Graphs

    Attributes
    ----------
    labels_ : np.ndarray
        Core value of each node.
    core_value_ : int
        Maximum core value of the graph

    Example
    -------
    >>> from sknetwork.topology import CoreDecomposition
    >>> from sknetwork.data import karate_club
    >>> kcore = CoreDecomposition()
    >>> adjacency = karate_club()
    >>> kcore.fit(adjacency)
    >>> kcore.core_value_
    4
    """
    def __init__(self):
        super(CoreDecomposition, self).__init__()
        self.labels_ = None
        self.core_value_ = None

    def fit(self, adjacency: sparse.csr_matrix) -> 'CoreDecomposition':
        """K-core decomposition.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.

        Returns
        -------
         self: :class:`CoreDecomposition`
        """
        labels = fit_core(adjacency.indptr, adjacency.indices)
        self.labels_ = labels
        self.core_value_ = labels.max()
        return self

    def fit_transform(self, adjacency: sparse.csr_matrix):
        """Fit algorithm to the data and return the core value of each node. Same parameters as the ``fit`` method.

        Returns
        -------
        labels :
            Core value of the nodes.
        """
        self.fit(adjacency)
        return self.labels_
