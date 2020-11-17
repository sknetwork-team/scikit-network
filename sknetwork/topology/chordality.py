#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 17, 2020
@author: Pierre Pebereau <pierre.pebereau@telecom-paris.fr>
@author: Alexis Barreaux <alexis.barreaux@telecom-paris.fr>
"""
from typing import Union

import numpy as np
import random as rd
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


def lexicographic_breadth_first_search(adjacency: Union[sparse.csr_matrix, np.ndarray]) -> list:
    """
    Sorts the vertices of a graph in lexicographic breadth-first search order.
    Parameters
    ----------
    adjacency: Union[sparse.csr_matrix, np.ndarray]
        Adjacency matrix of the graph.

    Returns
    -------
    numbers: int list
        The numbers assigned to each vertex with the number of vertex i in numbers[i].
    """
    n = adjacency.indptr.shape[0] - 1
    labels = [[] for _ in range(n)]
    numbers = [-1 for _ in range(n)]

    for i in range(n - 1, -1, -1):
        if i == n - 1:
            biggest_label_vertex = rd.randint(0, n - 1)
            print(biggest_label_vertex)
        else:
            unumbered = [i for i in range(n) if numbers[i] < 0]
            biggest_label_vertex = unumbered[0]
            for u in unumbered:
                if labels[u] > labels[biggest_label_vertex]:
                    biggest_label_vertex = u
        numbers[biggest_label_vertex] = i

        # Adding i to the labels of unumbered adjacent vertices.
        for j in adjacency.indices[adjacency.indptr[biggest_label_vertex]:adjacency.indptr[biggest_label_vertex + 1]]:
            if numbers[j] < 0:
                labels[j].append(str(i))

    return numbers
