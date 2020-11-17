#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 17, 2020
@author: Pierre Pebereau <pierre.pebereau@telecom-paris.fr>
@author: Alexis Barreaux <alexis.barreaux@telecom-paris.fr>
"""
from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.utils.base import Algorithm


class ChordalityTest(Algorithm):
    """Class to test if a graph is chordal or not..

    Parameters
    ----------

    Attributes
    ----------

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
 ,
    * Yannakakis, M., Tarjan, R. E. (1984)
      `Simple linear-time algorithms to test chordality of graphs, test acyclicity of hypergraphs,
       and selectively reduce acyclic hypergraphs.
       SIAM J. Comput., 13, pp. 566–579._

    """

    def __init__(self):
        pass

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'ChordalityTest':
        """Fit algorithm to the data.

        Parameters
        ----------
        adjacency : Union[sparse.csr_matrix, np.ndarray]
            Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`ChordalityTest
        """

    def fit_transform(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> np.ndarray:
        """Fit algorithm to the data and return the labels. Same parameters as the ``fit`` method.

        Returns
        -------
        labels : np.ndarray
            Labels.
        """
        self.fit(adjacency)
        return np.zeros(10)
