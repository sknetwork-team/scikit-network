# distutils: language = c++
# cython: language_level=3

"""
Created on June 19, 2020
@author: Pierre Pebereau <pierre.pebereau@telecom-paris.fr>
@author: Alexis Barreaux <alexis.barreaux@telecom-paris.fr>
"""

from typing import Union

import numpy as np
cimport numpy as np
from scipy import sparse

from sknetwork.utils.base import Algorithm
from sknetwork.utils.counting_sort cimport counting_sort

from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libc.math cimport log10 as clog10
from libc.math cimport pow as cpowl
from libc.math cimport modf
cimport cython

ctypedef pair[long long, int] cpair

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[long long, ndim=1] c_wl_coloring(np.ndarray[int, ndim=1] indices, np.ndarray[int, ndim=1] indptr, int max_iter, np.ndarray[int, ndim=1] input_labels) :
    DTYPE = np.int32
    cdef int n = indptr.shape[0] - 1
    cdef int iteration = 1
    cdef int u = 0
    cdef int j = 0
    cdef int current_max = 0
    cdef int i
    cdef int jj
    cdef int j1
    cdef int j2
    cdef int ind
    cdef int key
    cdef int deg
    cdef int max_deg
    cdef int neighbor_label
    cdef int old_label
    cdef long concatenation
    cdef double temp_conc
    cdef double int_part
    cdef bint has_changed

    cdef dict new_hash
    cdef long long[:] labels
    cdef long long[:] multiset
    cdef int[:]  degres
    cdef vector[cpair] large_label

    degres = memoryview(np.array(indptr[1:]) - np.array(indptr[:n]))
    max_deg = np.max(list(degres))

    cdef np.int32_t[:] count
    cdef long long[:] sorted_multiset = np.empty(max_deg, dtype=np.longlong)

    count= np.zeros(n, dtype = np.int32)
    multiset = np.empty(max_deg, dtype=np.longlong)
    labels = np.ones(n, dtype = np.longlong)
    large_label = np.zeros((n, 2), dtype=DTYPE)

    if max_iter > 0 :
        max_iter = min(n, max_iter)
    else :
        max_iter = n
    while iteration < max_iter :
        large_label.clear()
        for i in range(n):
            # 1
            # going through the neighbors of v.
            j = 0
            deg = degres[i]
            j1 = indptr[i]
            j2 = indptr[i + 1]
            for jj in range(j1,j2):
                u = indices[jj]
                multiset[j] = labels[u]
                j+=1

            # 2

            counting_sort(n, deg, count, multiset, sorted_multiset)
            concatenation = labels[i]
            for j in range(deg) :
                neighbor_label = multiset[j]

                temp_conc = clog10(neighbor_label)
                ret = modf(temp_conc, &int_part)
                int_part+=1.0
                temp_conc = cpowl(10.0, int_part)
                ret = modf(temp_conc, &int_part)
                concatenation= (concatenation *(<long>int_part)) + neighbor_label #there are still warnings because of np.int length

            large_label.push_back((cpair(concatenation, i)))


        # 3

        #TODO le problème est dans le argsort ici qui bouge les deux colonnes donc change l'ordre ensuite
        #large_label = large_label[large_label[:,0].argsort()] #.sort(key=lambda x: x[0])  # sort along first axis
        new_hash = {}
        current_max = 1

        has_changed = False #True if at least one label was changed
        for j in range(n):
            key = large_label[j].first
            ind = large_label[j].second
            if not (key in new_hash):
                new_hash[key] = current_max
                current_max += 1
            #  4
            old_label = int(labels[ind])
            labels[ind] = new_hash[key]

            has_changed += (old_label != labels[ind])
        iteration += 1


    print("iterations :", iteration)
    """
    lists = np.array([[0,[]] for _ in range(n)])
    for i in range(n) :
        lists[labels[i]][0] += 1
        lists[labels[i]][1].append(i)

    lists = lists[lists[:,0].argsort()]

    cdef int max_val = 0
    for i in range(n):
        j = len(lists[i][1])
        if j > 0 :
            for u in range(j):
                labels[lists[i][1][u]] = max_val
            max_val += 1
    """
    return np.asarray(labels)


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
    array([1, 2, 0, 0, 2])

    References
    ----------
    * Douglas, B. L. (2011).
      'The Weisfeiler-Lehman Method and Graph Isomorphism Testing.
      <https://arxiv.org/pdf/1101.5211.pdf>`_
      Cornell University.


    * Shervashidze, N., Schweitzer, P., van Leeuwen, E. J., Melhorn, K., Borgwardt, K. M. (2010)
      'Weisfeiler-Lehman graph kernels.
      <https://people.mpi-inf.mpg.de/~mehlhorn/ftp/genWLpaper.pdf>`_
      Journal of Machine Learning Research 1, 2010.
    """

    def __init__(self):
        super(WLColoring, self).__init__()

        self.labels_ = None

    def fit(self, int max_iter, adjacency: Union[sparse.csr_matrix, np.ndarray], input_labels : Union[sparse.csr_matrix, np.ndarray] = None) -> 'WLColoring':
        """Fit algorithm to the data.

        Parameters
        ----------
        max_iter : int
            Maximum number of iterations.

        adjacency : Union[sparse.csr_matrix, np.ndarray]
            Adjacency matrix of the graph.

        input_labels : Union[sparse.csr_matrix, np.ndarray]
            Input labels if the user wants to start with a specific input state.

        Returns
        -------
        self: :class:`WLColoring`
        """
        #TODO fin du PAF: remettre max_iter en attribut.
        indices = adjacency.indices
        indptr = adjacency.indptr

        self.labels_ = c_wl_coloring(indices, indptr, max_iter, input_labels)

        return self

    def fit_transform(self, int max_iter, adjacency: Union[sparse.csr_matrix, np.ndarray], input_labels : Union[sparse.csr_matrix, np.ndarray] = None) -> np.ndarray:
        """Fit algorithm to the data and return the labels. Same parameters as the ``fit`` method.

        Returns
        -------
        labels : np.ndarray
            Labels.
        """
        self.fit(max_iter, adjacency, input_labels)
        return np.asarray(self.labels_)
