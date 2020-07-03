# distutils: language = c++
# cython: language_level=3
"""
Created on July 2, 2020
@author: Pierre Pebereau <pierre.pebereau@telecom-paris.fr>
@author: Alexis Barreaux <alexis.barreaux@telecom-paris.fr>
"""
from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.topology.wl_core import c_wl_coloring, c_wl_kernel
from sknetwork.utils.base import Algorithm


class WLColoring(Algorithm):
    """Weisefeler-Lehman algorithm for coloring/labeling graphs in order to check similarity.

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations. -1 means infinity.

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
    >>> labels
    array([2, 0, 1, 1, 0], dtype=int32)

    References
    ----------
    * Douglas, B. L. (2011).
      `The Weisfeiler-Lehman Method and Graph Isomorphism Testing.<https://arxiv.org/pdf/1101.5211.pdf>̀_.

    * Shervashidze, N., Schweitzer, P., van Leeuwen, E. J., Melhorn, K., Borgwardt, K. M. (2011)
      `Weisfeiler-Lehman graph kernels.
      <http://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf?fbclid=IwAR2l9LJLq2VDfjT4E0ainE2p5dOxt\
      Be89gfSZJoYe4zi5wtuE9RVgzMKmFY>̀_
      Journal of Machine Learning Research 12, 2011.
    """
    def __init__(self, max_iter: int = -1):
        super(WLColoring, self).__init__()
        self.max_iter = max_iter
        self.labels_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'WLColoring':
        """Fit algorithm to the data.

        Parameters
        ----------
        adjacency : Union[sparse.csr_matrix, np.ndarray]
            Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`WLColoring`
        """
        n: int = adjacency.shape[0]
        if n < 3:
            self.labels_ = np.zeros(n)
        else:
            labels = np.ones(n, dtype=np.longlong)
            powers = np.ones(n, dtype=np.double)
            powers[1] = -np.pi / 3.15
            if self.max_iter > 0:
                max_iter = min(n, self.max_iter)
            else:
                max_iter = n
            indptr = adjacency.indptr.astype(np.int32)
            indices = adjacency.indices.astype(np.int32)
            self.labels_ = np.asarray(c_wl_coloring(indices, indptr, max_iter, labels, powers)[0]).astype(np.int32)
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


def wl_kernel(adjacency_1: Union[sparse.csr_matrix, np.ndarray], adjacency_2: Union[sparse.csr_matrix, np.ndarray],
              kernel_type: str = "subtree", n_iter: int = -1):
    """Graph kernels based on Weisefeler-Lehman coloring.

    Parameters
    -----------
    adjacency_1 : Union[sparse.csr_matrix, np.ndarray]
        First adjacency matrix.
    adjacency_2 : Union[sparse.csr_matrix, np.ndarray]
        Second adjacency matrix.
    kernel_type : str
        Kernel to use.

        * ``'isomorphism'``: checks if each graph has the same number of each distinct labels at each step.
        * ``'subtree'``: counts the number of occurences of each label in each graph at each step and sums the dot
          product of these counts.
        * ``'edge'``: counts the number of edges having the same labels for its nodes in both graphs at each step.

    n_iter : int
        Maximum number of iterations. Maximum positive value is the number of nodes in adjacency_1,
        it is also the default value set if given a negative int.

    Returns
    -------
    similarity : int
        Similarity score between graphs.

    Example
    -------
    >>> from sknetwork.topology import wl_kernel
    >>> from sknetwork.data import house
    >>> adjacency_1 = house()
    >>> adjacency_2 = house()
    >>> wl_kernel(adjacency_1, adjacency_2, "subtree")
    45

    References
    ----------
    * Douglas, B. L. (2011).
      `The Weisfeiler-Lehman Method and Graph Isomorphism Testing.<https://arxiv.org/pdf/1101.5211.pdf>̀_.

    * Shervashidze, N., Schweitzer, P., van Leeuwen, E. J., Melhorn, K., Borgwardt, K. M. (2011)
      `Weisfeiler-Lehman graph kernels.
      <http://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf?fbclid=IwAR2l9LJLq2VDfjT4E0ainE2p5dOxt\
      Be89gfSZJoYe4zi5wtuE9RVgzMKmFY>̀_
      Journal of Machine Learning Research 12, 2011.
    """
    kernels = {"isomorphism": 1, "subtree": 2, "edge": 3}
    try:
        return c_wl_kernel(adjacency_1, adjacency_2, kernels[kernel_type], n_iter)
    except KeyError:
        raise ValueError('Unknown kernel.')


def wl_similarity(adjacency1, adjacency2) -> float:
    """Normalized similarity between two graphs.

    :math:`\\dfrac{K(A,B)^2}{K(A,A)K(B,B)}̀.

    Where :math:`K` is the Weisfeiler-Lehman subtree kernel.

    Parameters
    ----------
    adjacency1 : Union[sparse.csr_matrix, np.ndarray]
        First graph to compare
    adjacency2 : Union[sparse.csr_matrix, np.ndarray]
        Second graph to compare

    Returns
    -------
    similarity : float
    """
    ab = wl_kernel(adjacency1, adjacency2, "subtree") ** 2
    aa = wl_kernel(adjacency1, adjacency1, "subtree")
    bb = wl_kernel(adjacency2, adjacency2, "subtree")
    return ab / (aa * bb)
