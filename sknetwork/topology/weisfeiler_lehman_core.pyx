# distutils: language = c++
# cython: language_level=3
"""
Created on July 1, 2020
@author: Pierre Pebereau <pierre.pebereau@telecom-paris.fr>
@author: Alexis Barreaux <alexis.barreaux@telecom-paris.fr>
"""
from libcpp.vector cimport vector
from libc.math cimport pow
cimport cython

ctypedef (int, double, int) ctuple

cdef extern from "<algorithm>" namespace "std":
    # fixed sort as per https://stackoverflow.com/questions/57584909/unable-to-use-cdef-function-in-stdsort-as-comparison-function
    void sort(...)
    void csort "sort"(...)

cdef bint is_lower(ctuple a, ctuple b) :
    """Lexicographic comparison between triplets based on the first two values.

    Parameters
    ----------
    a:
        First triplet.
    b:
        Second triplet.

    Returns
    -------
    ``True`` if a < b, and ``False`` otherwise.
    """
    cdef int a1, b1
    cdef double a2, b2

    a1, a2, _ = a
    b1, b2, _ = b
    if a1 == b1 :
        return a2 < b2
    return a1 < b1


@cython.boundscheck(False)
@cython.wraparound(False)
def weisfeiler_lehman_coloring(int[:] indptr, int[:] indices, int[:] labels, double [:] powers, int max_iter):
    """Weisfeiler-Lehman coloring.

    Parameters
    ----------
    indptr :
        Indptr of the CSR.
    indices :
        Indices of the CSR.
    labels : int[:]
        Labels to be changed.
    powers : double [:]
        Powers being used as hash and put in a memory view to limit several identical calculations.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    labels : int[:]
        Colors of the nodes.
    has_changed : bint
        True if the output labels are not the same as the input ones. False otherwise.
    """
    cdef int n = indptr.shape[0] -1
    cdef int iteration = 0
    cdef int i, j, j1, j2, jj, label

    cdef double epsilon = pow(10, -10)

    cdef vector[ctuple] new_labels
    cdef ctuple tuple_ref, tuple_new
    cdef double hash_ref, hash_new
    cdef int label_ref, label_new

    cdef bint has_changed = True

    while iteration < max_iter and has_changed:
        new_labels.clear()
        has_changed = False

        for i in range(n):
            hash_ref = 0
            j1 = indptr[i]
            j2 = indptr[i + 1]

            for jj in range(j1, j2):
                j = indices[jj]
                hash_ref += powers[labels[j]]

            new_labels.push_back((labels[i], hash_ref, i))
        csort(new_labels.begin(), new_labels.end(), is_lower)
        label = 0
        tuple_new = new_labels[0]
        labels[tuple_new[2]] = label
        for j in range(1, n):
            tuple_ref = tuple_new
            tuple_new = new_labels[j]
            label_ref, hash_ref, _ = tuple_ref
            label_new, hash_new, i = tuple_new

            if abs(hash_new - hash_ref) > epsilon or label_new != label_ref :
                label += 1

            if labels[i] != label:
                has_changed = True

            labels[i] = label
        iteration += 1

    return labels, has_changed
