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

cdef fit_core(int n_nodes, int[:] indptr, int[:] indices):
    """	   Orders the nodes of the graph according to their core value.

    Parameters
    ----------
    n_nodes :
        Number of nodes.
    indptr :
        CSR format index array of the normalized adjacency matrix.
    indices :
        CSR format index pointer array of the normalized adjacency matrix.

    Returns
    -------
    cores :
        Numpy array of the nodes ordered in descending order by their core value.
    core_value :
        Maximum core value
    """

    cdef int core_value		# current/max core value of the graph
    cdef int min_node		# current node of minimum degree
    cdef int i, j, k
    cdef int[:] degrees						# array of each node degrees
    cdef np.ndarray[int, ndim=1] k_cores	# array of ordered nodes

    cdef MinHeap mh			# minimum heap with an update system

    degrees = np.empty((n_nodes,), dtype=np.int32)

    for i in range(n_nodes):
        degrees[i] = indptr[i+1] - indptr[i]

    # creates an heap of sufficient size to contain all nodes
    mh = MinHeap.__new__(MinHeap, n_nodes)

    # inserts all nodes in the heap
    for i in range(n_nodes):
        mh.insertKey(i, degrees)

    i = n_nodes - 1		# index of the rear of the list/array
    core_value = 0

    k_cores = np.empty((n_nodes,), dtype=np.int32)		# initializes an empty array

    while not mh.isEmpty():		# until the heap is emptied
        min_node = mh.extractMin(degrees)
        core_value = max(core_value, degrees[min_node])

        # decreases the degree of each neighbors of min_node to simulate its deletion
        for k in range(indptr[min_node], indptr[min_node+1]):
            j = indices[k]
            degrees[j] -= 1
            mh.decreaseKey(j, degrees)		# updates the heap to take into account the new degrees

        k_cores[i] = min_node	# insert the node of minimum degree at the end of the array
        i -= 1

    return k_cores, core_value


class CoreDecomposition:
    """ k-core Decomposition algorithm.

    * Graphs

    Attributes
    ----------
    ordered : np.ndarray
        Nodes sorted by their core value in decreasing order
    core_value : int
        Maximum core value of the graph

    Example
    -------
    >>> from sknetwork.topology import CoreDecomposition
    >>> from sknetwork.data import karate_club
    >>> core = CoreDecomposition()
    >>> graph = karate_club()
    >>> adjacency = graph.adjacency
    >>> core.fit_transform(adjacency)
    4
    """

    def __init__(self):
        self.ordered = None
        self.core_value = 0

    def fit(self, adjacency : sparse.csr_matrix) -> 'CoreDecomposition':
        """ k-core decomposition.

        Parameters
        ----------
        adjacency:
            Adjacency matrix of the graph.

        Returns
        -------
         self: :class:`CoreDecomposition`
        """
        cores, val = fit_core(adjacency.shape[0], adjacency.indptr, adjacency.indices)
        self.ordered = cores
        self.core_value = val
        return self

    def fit_transform(self, *args, **kwargs) -> int:
        """ Fit algorithm to the data and return the maximum core value. Same parameters as the ``fit`` method.

        Returns
        -------
        core_value : int
            Maximum core value
        """
        self.fit(*args, **kwargs)
        return self.core_value
