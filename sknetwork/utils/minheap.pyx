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
