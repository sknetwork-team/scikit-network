#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in July 2020
@author: Pierre Pebereau <pierre.pebereau@telecom-paris.fr>
@author: Alexis Barreaux <alexis.barreaux@telecom-paris.fr>
"""
from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.topology.weisfeiler_lehman_core import weisfeiler_lehman_coloring
from sknetwork.utils.check import check_format, check_square


def color_weisfeiler_lehman(adjacency: Union[sparse.csr_matrix, np.ndarray], max_iter: int = -1) -> np.ndarray:
    """Color nodes using Weisfeiler-Lehman algorithm.

    Parameters
    ----------
    adjacency : sparse.csr_matrix
        Adjacency matrix of the graph
    max_iter : int
        Maximum number of iterations. Negative value means no limit (until convergence).

    Returns
    -------
    labels : np.ndarray
        Label of each node.

    Example
    -------
    >>> from sknetwork.data import house
    >>> adjacency = house()
    >>> labels = color_weisfeiler_lehman(adjacency)
    >>> print(labels)
    [0 2 1 1 2]

    References
    ----------
    * Douglas, B. L. (2011).
      `The Weisfeiler-Lehman Method and Graph Isomorphism Testing.
      <https://arxiv.org/pdf/1101.5211.pdf>`_

    * Shervashidze, N., Schweitzer, P., van Leeuwen, E. J., Melhorn, K., Borgwardt, K. M. (2011)
      `Weisfeiler-Lehman graph kernels.
      <https://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf>`_
      Journal of Machine Learning Research 12, 2011.
    """

    adjacency = check_format(adjacency, allow_empty=True)
    check_square(adjacency)
    n_nodes = adjacency.shape[0]
    if max_iter < 0 or max_iter > n_nodes:
        max_iter = n_nodes

    labels = np.zeros(n_nodes, dtype=np.int32)
    powers = (-np.pi / 3.15) ** np.arange(n_nodes, dtype=np.double)
    indptr = adjacency.indptr
    indices = adjacency.indices
    labels, _ = weisfeiler_lehman_coloring(indptr, indices, labels, powers, max_iter)
    return np.array(labels)


def are_isomorphic(adjacency1: sparse.csr_matrix, adjacency2: sparse.csr_matrix, max_iter: int = -1) -> bool:
    """Weisfeiler-Lehman isomorphism test. If the test is False, the graphs cannot be isomorphic.

    Parameters
    -----------
    adjacency1 :
        First adjacency matrix.
    adjacency2 :
        Second adjacency matrix.
    max_iter : int
        Maximum number of iterations. Negative value means no limit (until convergence).

    Returns
    -------
    test_result : bool

    Example
    -------
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
      <https://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf>`_
      Journal of Machine Learning Research 12, 2011.
    """
    adjacency1 = check_format(adjacency1)
    check_square(adjacency1)
    adjacency2 = check_format(adjacency2)
    check_square(adjacency2)

    if (adjacency1.shape != adjacency2.shape) or (adjacency1.nnz != adjacency2.nnz):
        return False

    n_nodes = adjacency1.shape[0]

    if max_iter < 0 or max_iter > n_nodes:
        max_iter = n_nodes

    indptr1 = adjacency1.indptr
    indptr2 = adjacency2.indptr
    indices1 = adjacency1.indices
    indices2 = adjacency2.indices

    labels1 = np.zeros(n_nodes, dtype=np.int32)
    labels2 = np.zeros(n_nodes, dtype=np.int32)

    powers = (-np.pi / 3.15) ** np.arange(n_nodes, dtype=np.double)

    iteration = 0
    has_changed1, has_changed2 = True, True
    while iteration < max_iter and (has_changed1 or has_changed2):
        labels1, has_changed1 = weisfeiler_lehman_coloring(indptr1, indices1, labels1, powers, max_iter=1)
        labels2, has_changed2 = weisfeiler_lehman_coloring(indptr2, indices2, labels2, powers, max_iter=1)
        _, counts1 = np.unique(np.array(labels1), return_counts=True)
        _, counts2 = np.unique(np.array(labels2), return_counts=True)
        if (counts1 != counts2).any():
            return False
        iteration += 1

    return True
