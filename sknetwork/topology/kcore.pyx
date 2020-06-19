# distutils: language = c++
# cython: language_level=3
"""
Created on Jun 3, 2020
@author: Julien Simonnet <julien.simonnet@etu.upmc.fr>
@author: Yohann Robert <yohann.robert@etu.upmc.fr>
"""
#========== Minimum heap ==========

cimport cython

import numpy as np
cimport numpy as np

from scipy import sparse

from sknetwork.utils.base import Algorithm


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class MinHeap:

    def __cinit__(self, int n):
        self.arr.reserve(n)		# reserves the necessary space in the vector
        self.pos.reserve(n)		# reserves the necessary space in the other vector
        self.size = 0

    cdef bint isEmpty(self):
        return self.size == 0

    cdef inline void swap(self, int x, int y):
        cdef int tmp
        tmp = self.arr[x]
        self.arr[x] = self.arr[y]
        self.arr[y] = tmp

        # updates the position of the corresponding elements
        self.pos[self.arr[x]] = x
        self.pos[self.arr[y]] = y

    # Inserts a new key k
    cdef void insertKey(self, int k, int[:] degrees):
        # First insert the new key at the end
        self.arr[self.size] = k
        self.pos[k] = self.size
        cdef int i = self.size
        self.size += 1

        cdef int p = parent(i)
        while (p >= 0) and (degrees[self.arr[p]] > degrees[self.arr[i]]) :
            self.swap(i, p)
            i = p
            p = parent(i)


    # Decreases value of key at index 'i' to new_val.  It is assumed that
    # the new value is smaller than the old one
    cdef void decreaseKey(self, int i, int[:] degrees):
        cdef int pos, p
        pos = self.pos[i]
        if pos < self.size:
            p = parent(pos)

            while (pos != 0) and (degrees[self.arr[p]] > degrees[self.arr[pos]]):
                self.swap(pos, p)
                pos = p

    # Function to remove minimum element (or root) from min heap
    cdef int extractMin(self, int[:] degrees):
        if self.size == 1:
            self.size = 0
            return self.arr[0]

        # Store the minimum value, and remove it from heap
        cdef int root = self.arr[0]
        self.arr[0] = self.arr[self.size-1]
        self.size -= 1
        self.minHeapify(0, degrees)

        return root

    # A recursive method to heapify a subtree with the root at given index
    # This function assumes that the subtrees are already heapified
    cdef void minHeapify(self, int i, int[:] degrees):
        cdef int l, r, smallest
        l = left(i)
        r = right(i)
        smallest = i
        if (l < self.size) and (degrees[self.arr[l]] < degrees[self.arr[i]]):
            smallest = l

        if (r < self.size) and (degrees[self.arr[r]] < degrees[self.arr[smallest]]):
            smallest = r

        if smallest != i:
            self.swap(i, smallest)
            self.minHeapify(smallest, degrees)


#========== k-core decomposition ==========

cdef fit_core(int[:] indptr, int[:] indices):
    """	   Orders the nodes of the graph according to their core value.

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
    cdef int core_value		# current/max core value of the graph
    cdef int min_node		# current node of minimum degree
    cdef int i, j, k
    cdef int[:] degrees						# array of each node degrees
    cdef np.ndarray[int, ndim=1] labels = np.empty((n,), dtype=np.int32)	# array of ordered nodes

    cdef MinHeap mh			# minimum heap with an update system

    degrees = np.asarray(indptr)[1:] - np.asarray(indptr)[:-1]

    # creates an heap of sufficient size to contain all nodes
    mh = MinHeap.__new__(MinHeap, n)

    # inserts all nodes in the heap
    for i in range(n):
        mh.insertKey(i, degrees)

    i = n - 1		# index of the rear of the list/array
    core_value = 0

    while not mh.isEmpty():		# until the heap is emptied
        min_node = mh.extractMin(degrees)
        core_value = max(core_value, degrees[min_node])

        # decreases the degree of each neighbors of min_node to simulate its deletion
        for k in range(indptr[min_node], indptr[min_node+1]):
            j = indices[k]
            degrees[j] -= 1
            mh.decreaseKey(j, degrees)		# updates the heap to take into account the new degrees

        labels[min_node] = core_value	# insert the node of minimum degree at the end of the array
        i -= 1

    return np.asarray(labels)


class CoreDecomposition(Algorithm):
    """ k-core Decomposition algorithm.

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
    >>> graph = karate_club()
    >>> adjacency = graph.adjacency
    >>> kcore.fit(adjacency)
    >>> kcore.core_value_
    4
    """
    def __init__(self):
        super(CoreDecomposition, self).__init__()
        self.labels_ = None
        self.core_value_ = None

    def fit(self, adjacency: sparse.csr_matrix) -> 'CoreDecomposition':
        """ k-core decomposition.

        Parameters
        ----------
        adjacency:
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
