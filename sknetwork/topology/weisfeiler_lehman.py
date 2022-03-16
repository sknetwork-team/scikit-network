#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 2, 2020
@author: Pierre Pebereau <pierre.pebereau@telecom-paris.fr>
@author: Alexis Barreaux <alexis.barreaux@telecom-paris.fr>
"""
from typing import Union

import numpy as np
from scipy import sparse
from sknetwork.topology.weisfeiler_lehman_core import weisfeiler_lehman_coloring

from sknetwork.utils.base import Algorithm


class WeisfeilerLehman(Algorithm):
    """Weisfeiler-Lehman algorithm for coloring/labeling graphs in order to check similarity.

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations. Negative value  means until convergence.

    Attributes
    ----------
    labels_ : np.ndarray
        Label of each node.

    Example
    -------
    >>> from sknetwork.topology import WeisfeilerLehman
    >>> from sknetwork.data import house
    >>> weisfeiler_lehman = WeisfeilerLehman()
    >>> adjacency = house()
    >>> labels = weisfeiler_lehman.fit_transform(adjacency)
    >>> labels
    array([0, 2, 1, 1, 2], dtype=int32)

    References
    ----------
    * Douglas, B. L. (2011).
      `The Weisfeiler-Lehman Method and Graph Isomorphism Testing.
      <https://arxiv.org/pdf/1101.5211.pdf>`_

    * Shervashidze, N., Schweitzer, P., van Leeuwen, E. J., Melhorn, K., Borgwardt, K. M. (2011)
      `Weisfeiler-Lehman graph kernels.
      <http://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf>`_
      Journal of Machine Learning Research 12, 2011.
    """
    def __init__(self, max_iter: int = -1):
        super(WeisfeilerLehman, self).__init__()
        self.max_iter = max_iter
        self.labels_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'WeisfeilerLehman':
        """Fit algorithm to the data.

        Parameters
        ----------
        adjacency : Union[sparse.csr_matrix, np.ndarray]
            Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`WeisfeilerLehman`
        """
        n: int = adjacency.shape[0]
        if self.max_iter < 0 or self.max_iter > n:
            max_iter = np.int32(n)
        else:
            max_iter = np.int32(self.max_iter)

        labels = np.zeros(n, dtype=np.int32)
        powers = (-np.pi / 3.15) ** np.arange(n, dtype=np.double)
        indptr = adjacency.indptr.astype(np.int32)
        indices = adjacency.indices.astype(np.int32)

        labels, _ = weisfeiler_lehman_coloring(indptr, indices, labels, powers, max_iter)
        self.labels_ = np.asarray(labels).astype(np.int32)
        return self

    def fit_transform(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> np.ndarray:
        """Fit algorithm to the data and return the labels. Same parameters as the ``fit`` method.

        Returns
        -------
        labels : np.ndarray
            Labels.
        """
        self.fit(adjacency)
        return self.labels_


def are_isomorphic(adjacency1: sparse.csr_matrix,
                   adjacency2: sparse.csr_matrix, max_iter: int = -1) -> bool:
    """Weisfeiler-Lehman isomorphism test. If the test is False, the graphs cannot be isomorphic,
    otherwise, they might be.

    Parameters
    -----------
    adjacency1 :
        First adjacency matrix.
    adjacency2 :
        Second adjacency matrix.
    max_iter : int
        Maximum number of coloring iterations. Negative value means until convergence.

    Returns
    -------
    test_result : bool

    Example
    -------
    >>> from sknetwork.topology import are_isomorphic
    >>> from sknetwork.data import house, bow_tie
    >>> are_isomorphic(house(), bow_tie())
    False

    References
    ----------
    * Douglas, B. L. (2011).
      `The Weisfeiler-Lehman Method and Graph Isomorphism Testing.
      <https://arxiv.org/pdf/1101.5211.pdf>`_

    * Shervashidze, N., Schweitzer, P., van Leeuwen, E. J., Melhorn, K., Borgwardt, K. M. (2011)
      `Weisfeiler-Lehman graph kernels.
      <http://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf>`_
      Journal of Machine Learning Research 12, 2011.
    """
    if (adjacency1.shape != adjacency2.shape) or (adjacency1.nnz != adjacency2.nnz):
        return False

    n = adjacency1.shape[0]

    if max_iter < 0 or max_iter > n:
        max_iter = n

    indptr1 = adjacency1.indptr.astype(np.int32)
    indptr2 = adjacency2.indptr.astype(np.int32)
    indices1 = adjacency1.indices.astype(np.int32)
    indices2 = adjacency2.indices.astype(np.int32)

    labels_1 = np.zeros(n, dtype=np.int32)
    labels_2 = np.zeros(n, dtype=np.int32)

    powers = (- np.pi / 3.15) ** np.arange(n, dtype=np.double)

    iteration = 0
    has_changed_1, has_changed_2 = True, True
    while iteration < max_iter and (has_changed_1 or has_changed_2):
        labels_1, has_changed_1 = weisfeiler_lehman_coloring(indptr1, indices1, labels_1, powers, max_iter=1)
        labels_2, has_changed_2 = weisfeiler_lehman_coloring(indptr2, indices2, labels_2, powers, max_iter=1)

        colors_1, counts_1 = np.unique(np.asarray(labels_1), return_counts=True)
        colors_2, counts_2 = np.unique(np.asarray(labels_2), return_counts=True)

        if (colors_1.shape != colors_2.shape) or (counts_1 != counts_2).any():
            return False

        iteration += 1

    return True
