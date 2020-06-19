# distutils: language = c++
# cython: language_level=3
"""
Created on Jun 3, 2020
@author: Julien Simonnet <julien.simonnet@etu.upmc.fr>
@author: Yohann Robert <yohann.robert@etu.upmc.fr>
"""
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class MinHeap:
    """Min heap data structure."""
    def __cinit__(self, int n):
        self.arr.reserve(n)		# reserves the necessary space in the vector
        self.pos.reserve(n)		# reserves the necessary space in the other vector
        self.size = 0

    cdef bint empty(self):
        """Check if the heap is empty."""
        return self.size == 0

    cdef inline void swap(self, int x, int y):
        """Exchange two elements in the heap."""
        cdef int tmp
        tmp = self.arr[x]
        self.arr[x] = self.arr[y]
        self.arr[y] = tmp

        # updates the position of the corresponding elements
        self.pos[self.arr[x]] = x
        self.pos[self.arr[y]] = y

    # Inserts a new key k
    cdef void insert_key(self, int k, int[:] degrees):
        """Insert new element into the heap"""
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


    cdef void decrease_key(self, int i, int[:] degrees):
        """Decrease value of key at index 'i' to new_val.
        It is assumed that the new value is smaller than the old one.
        """
        cdef int pos, p
        pos = self.pos[i]
        if pos < self.size:
            p = parent(pos)

            while (pos != 0) and (degrees[self.arr[p]] > degrees[self.arr[pos]]):
                self.swap(pos, p)
                pos = p

    cdef int pop_min(self, int[:] degrees):
        """Remove and return the minimum element (or root) from the heap."""
        if self.size == 1:
            self.size = 0
            return self.arr[0]

        # Store the minimum value, and remove it from heap
        cdef int root = self.arr[0]
        self.arr[0] = self.arr[self.size-1]
        self.size -= 1
        self.min_heapify(0, degrees)

        return root

    cdef void min_heapify(self, int i, int[:] degrees):
        """A recursive method to heapify a subtree with the root at given index
        This function assumes that the subtrees are already heapified.
        """
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
            self.min_heapify(smallest, degrees)
