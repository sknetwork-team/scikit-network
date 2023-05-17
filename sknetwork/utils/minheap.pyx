# distutils: language = c++
# cython: language_level=3
"""
Created in June 2020
@author: Julien Simonnet <julien.simonnet@etu.upmc.fr>
@author: Yohann Robert <yohann.robert@etu.upmc.fr>
"""
cimport cython


cdef inline int parent(int i):
    """Index of the parent node of i in the tree."""
    return (i - 1) // 2

cdef inline int left(int i):
    """Index of the left child of i in the tree."""
    return 2 * i + 1

cdef inline int right(int i):
    """Index of the right child of i in the tree."""
    return 2 * i + 2


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class MinHeap:
    """Min heap data structure."""
    def __cinit__(self, int n):
        self.val.reserve(n)		# reserves the necessary space in the vector
        self.pos.reserve(n)		# reserves the necessary space in the other vector
        self.size = 0

    cdef bint empty(self):
        """Check if the heap is empty."""
        return self.size == 0

    cdef inline void swap(self, int x, int y):
        """Exchange two elements in the heap."""
        cdef int tmp
        tmp = self.val[x]
        self.val[x] = self.val[y]
        self.val[y] = tmp

        # updates the position of the corresponding elements
        self.pos[self.val[x]] = x
        self.pos[self.val[y]] = y

    # Inserts a new key k
    cdef void insert_key(self, int k, int[:] scores):
        """Insert new element into the heap"""
        # First insert the new key at the end
        self.val[self.size] = k
        self.pos[k] = self.size
        cdef int i = self.size
        self.size += 1

        cdef int p = parent(i)
        while (p >= 0) and (scores[self.val[p]] > scores[self.val[i]]) :
            self.swap(i, p)
            i = p
            p = parent(i)


    cdef void decrease_key(self, int i, int[:] scores):
        """Decrease value of key at index 'i' to new_val.
        It is assumed that the new value is smaller than the old one.
        """
        cdef int pos, p
        pos = self.pos[i]
        if pos < self.size:
            p = parent(pos)

            while (pos != 0) and (scores[self.val[p]] > scores[self.val[pos]]):
                self.swap(pos, p)
                pos = p
                p = parent(pos)

    cdef int pop_min(self, int[:] scores):
        """Remove and return the minimum element (or root) from the heap."""
        if self.size == 1:
            self.size = 0
            return self.val[0]

        # Store the minimum value, and remove it from heap
        cdef int root = self.val[0]
        self.val[0] = self.val[self.size-1]
        self.pos[self.val[0]] = 0
        self.size -= 1
        self.min_heapify(0, scores)

        return root

    cdef void min_heapify(self, int i, int[:] scores):
        """A recursive method to heapify a subtree with the root at given index
        This function assumes that the subtrees are already heapified.
        """
        cdef int l, r, smallest
        l = left(i)
        r = right(i)
        smallest = i
        if (l < self.size) and (scores[self.val[l]] < scores[self.val[i]]):
            smallest = l

        if (r < self.size) and (scores[self.val[r]] < scores[self.val[smallest]]):
            smallest = r

        if smallest != i:
            self.swap(i, smallest)
            self.min_heapify(smallest, scores)
