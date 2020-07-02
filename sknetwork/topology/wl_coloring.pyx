# distutils: language = c++
# cython: language_level=3

"""
Created on July 2, 2020
@author: Pierre Pebereau <pierre.pebereau@telecom-paris.fr>
@author: Alexis Barreaux <alexis.barreaux@telecom-paris.fr>
"""

from typing import Union

import numpy as np
cimport numpy as np
from scipy import sparse

from sknetwork.utils.base import Algorithm

from libcpp.vector cimport vector
from libcpp.algorithm cimport sort as csort
from libc.math cimport pow as cpowl
cimport cython

cdef bint is_lower(ctuple p1,ctuple p2) :
    cdef long long p11, p21
    cdef double p12, p22
    cdef int p13, p23

    p11, p12, p13 = p1
    p21, p22, p23 = p2
    if p11 == p21 :
        return p12 < p22
    return p11 < p21

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef long long[:] wl_coloring(adjacency : Union[sparse.csr_matrix, np.ndarray], int max_iter) :
    """Wrapper for Weifeiler-Lehman Coloring
    Parameters
    ----------
    adjacency : Union[sparse.csr_matrix, np.ndarray]
        Adjacency matrix of the graph to color (expected to be in CSR format).

    max_iter : int
        Maximum number of iterations once wants to make. Maximum positive value is the number of nodes in adjacency_1,
        it is also the default value set if given a negative int.

    Returns
    -------
    labels : long long[:]
        Memory view made of long long being the labels for the coloring.
    """

    cdef int n = adjacency.indptr.shape[0]-1
    cdef double alpha = - np.pi/3.15

    cdef long long[:] labels = np.ones(n, dtype = np.longlong)

    cdef double [:] powers = np.ones(n, dtype = np.double)

    cdef int k
    for k in range(1,n) :
        powers[k] = powers[k-1]*alpha


    if max_iter > 0 :
        max_iter = min(n, max_iter)
    else :
        max_iter = n

    c_wl_coloring(adjacency.indices, adjacency.indptr, max_iter, labels, powers)

    return labels

@cython.boundscheck(False)
@cython.wraparound(False)
cdef (int, bint) c_wl_coloring(np.ndarray[int, ndim=1] indices,
                               np.ndarray[int, ndim=1] indptr,
                               int max_iter,
                               long long[:] labels,
                               double [:] powers):
    """ Weifeiler-Lehman inspired coloring.

    Parameters
    ----------
    indices : np.ndarray[int, ndim=1]
        Indices of the graph in CSR format.

    indptr : np.ndarray[int, ndim=1]
        Indptr of the second graph in CSR format.

    max_iter : int
        Maximum number of iterations once wants to make.

    labels : long long[:]
        Labels to be changed.

    powers : double [:]
        Powers being used as hash and put in a memory view to limit several identical calculations.


    Returns
    -------
    current_max : int
        Used in wl_kernel to limit a loop.

    has_changed : bint
        Used in wl_kernel to limit a loop.
    """
    cdef int iteration, i, j, j1, j2, jj, u, current_max
    cdef double temp_pow
    cdef int n = indptr.shape[0] -1

    cdef double epsilon

    cdef vector[ctuple] new_labels
    cdef double current_hash
    cdef ctuple current_tuple
    cdef ctuple previous_tuple

    epsilon = cpowl(10, -10)

    cdef long long current_label, previous_label
    cdef double previous_hash
    cdef int current_vertex, previous_vertex
    cdef bint has_changed

    iteration = 0
    has_changed = True

    while iteration <= max_iter and has_changed:
        new_labels.clear()
        for i in range(n):# going through the neighbors of v.
            current_hash = 0
            j1 = indptr[i]
            j2 = indptr[i + 1]
            for jj in range(j1,j2):
                u = indices[jj]

                current_hash += powers[labels[u]]
            new_labels.push_back((labels[i], current_hash, i))
        csort(new_labels.begin(), new_labels.end(),is_lower)

        current_max = 0
        current_tuple = new_labels[0]
        current_label, current_hash, current_vertex = current_tuple
        labels[current_vertex] = current_max
        has_changed = False
        for jj in range(1,n):
            previous_tuple = current_tuple
            current_tuple = new_labels[jj]
            previous_label, previous_hash, previous_vertex = previous_tuple
            current_label, current_hash, current_vertex = current_tuple
            if abs(previous_hash - current_hash) > epsilon or previous_label != current_label :
                current_max+=1

            if not has_changed:
                has_changed = labels[current_vertex] != current_max

            labels[current_vertex] = current_max

        iteration+=1

    return current_max, has_changed

class WLColoring(Algorithm):
    """Weisefeler-Lehman algorithm for coloring/labeling graphs in order to check similarity.

    Attributes
    ----------
    labels_ : np.ndarray
        Label of each node.

    Example
    -------
    >>> from sknetwork.topology import WLColoring
    >>> from sknetwork.data import house
    >>> wlcoloring = WLColoring()
    >>> adjacency = house()
    >>> labels = wlcoloring.fit_transform(adjacency)
    array([2, 0, 1, 1, 0])

    References
    ----------
    * Douglas, B. L. (2011).
      'The Weisfeiler-Lehman Method and Graph Isomorphism Testing.
      <https://arxiv.org/pdf/1101.5211.pdf>`_
      Cornell University.


    * Shervashidze, N., Schweitzer, P., van Leeuwen, E. J., Melhorn, K., Borgwardt, K. M. (2011)
      'Weisfeiler-Lehman graph kernels.
      <http://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf?fbclid=IwAR2l9LJLq2VDfjT4E0ainE2p5dOxtBe89gfSZJoYe4zi5wtuE9RVgzMKmFY>`_
      Journal of Machine Learning Research 12, 2011.
    """

    def __init__(self):
        super(WLColoring, self).__init__()

        self.labels_ = None

    def fit(self, int max_iter, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'WLColoring':
        """Fit algorithm to the data.

        Parameters
        ----------
        max_iter : int
            Maximum number of iterations.

        adjacency : Union[sparse.csr_matrix, np.ndarray]
            Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`WLColoring`
        """

        self.labels_ = np.asarray(wl_coloring(adjacency, max_iter))

        return self

    def fit_transform(self, int max_iter, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> np.ndarray:
        """Fit algorithm to the data and return the labels. Same parameters as the ``fit`` method.

        Returns
        -------
        labels : np.ndarray
            Labels.
        """
        self.fit(max_iter, adjacency)
        return self.labels_
