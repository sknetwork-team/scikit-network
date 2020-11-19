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


def lexicographic_breadth_first_search_v2(adjacency: Union[sparse.csr_matrix, np.ndarray]) -> list:
    """
    Sorts the vertices of a graph in lexicographic breadth-first search order.
    Parameters
    ----------
    adjacency: Union[sparse.csr_matrix, np.ndarray]
        Adjacency matrix of the graph.

    Returns
    -------
    lex_order: int list
        The vertices sorted in lexicographic bread-first search order. lex_order[i] contains the i-th vertex in this
        order.
    """
    # TODO : j'ai inversé le sens de l'algo sur wiki, j'ai placé les plus longs labels au bout.
    n = adjacency.indptr.shape[0] - 1
    lex_order = [-1 for _ in range(n)]

    vertices_sets = [[i for i in range(n)]]

    for i in range(n - 1, -1, -1):
        cur_vertex = vertices_sets[-1].pop()
        if len(vertices_sets[-1]) == 0:
            vertices_sets.pop()

        # Assigning number to cur_vertex.
        lex_order[i] = cur_vertex
        # Searching for neighbors of cur_vertex.
        cur_neighbors = adjacency.indices[adjacency.indptr[cur_vertex]:adjacency.indptr[cur_vertex + 1]]
        count = 0  # The position on which to add the next new set.

        while count < len(vertices_sets):
            vset = vertices_sets[count]
            count += 1
            new_set = []

            # Creating new set.
            for candidate in vset:
                if candidate in cur_neighbors:
                    new_set.append(candidate)

            # Updating old set and list of sets.
            if len(new_set) > 0:
                for vertex in new_set:
                    vset.remove(vertex)
                if len(vset) == 0:
                    count -= 1
                    vertices_sets.pop(count)
                vertices_sets.insert(count, new_set)
                count += 1

    return lex_order


def lexicographic_breadth_first_search(adjacency: Union[sparse.csr_matrix, np.ndarray]) -> list:
    """
    Sorts the vertices of a graph in lexicographic breadth-first search order.
    Parameters
    ----------
    adjacency: Union[sparse.csr_matrix, np.ndarray]
        Adjacency matrix of the graph.

    Returns
    -------
    lex_order: int list
        The vertices sorted in the opposite of a lexicographic bread-first search order. lex_order[i] contains the i-th
        vertex in this order.
    """
    n = adjacency.indptr.shape[0] - 1
    labels = [[] for _ in range(n)]
    position = [-1 for _ in range(n)]

    for i in range(n - 1, -1, -1):
        if i == n - 1:
            biggest_label_vertex = n - 1
        else:
            unnumbered = [v for v in range(n) if position[v] < 0]
            biggest_label_vertex = unnumbered[0]
            for u in unnumbered:
                if labels[u] >= labels[biggest_label_vertex]:
                    biggest_label_vertex = u
        print(labels)
        position[biggest_label_vertex] = i

        # Adding i to the labels of unnumbered adjacent vertices.
        for j in adjacency.indices[adjacency.indptr[biggest_label_vertex]:adjacency.indptr[biggest_label_vertex + 1]]:
            if position[j] < 0:
                labels[j].append(str(i))

    lex_order = [0 for _ in range(n)]
    for i in range(n):
        lex_order[position[i]] = i

    return lex_order


def is_chordal(adjacency: Union[sparse.csr_matrix, np.ndarray]) -> bool:
    """
    Takes the adjacency matrix of a graph and tells if it is chordal or not.
    Parameters
    ----------
    adjacency: Union[sparse.csr_matrix, np.ndarray]
        Adjacency matrix of the graph.

    Returns
    -------
    result: bool
        A boolean stating wether this graph is chordal or not.
    """
    lex_order = lexicographic_breadth_first_search(adjacency)

    n = adjacency.indptr.shape[0] - 1

    if n <= 3:
        # Any graph with at most 3 vertices is chordal.
        return True

    # We must check if for any vertex, his neighbors other than the closest one are also neighbors of his closest
    # neighbor. Said in another manner, we must check if for any vertex, him and his neighbors found after him in the
    # sorting make a clique.

    for i in range(0, n - 2):  # We can stop before the two last vertex since two neighbors form a clique.

        vertex = lex_order[i]
        neighbors = adjacency.indices[adjacency.indptr[vertex]: adjacency.indptr[vertex + 1]]

        # Searching for a neighbor of the current vertex placed after him and as close as possible in the sorting.
        closest_neighbor = -1
        pos_closest = - 1
        for j in range(i + 1, n):
            if lex_order[j] in neighbors:
                closest_neighbor = lex_order[j]
                pos_closest = j
                break

        if closest_neighbor < 0:
            continue
        else:
            closest_neighbors = adjacency.indices[adjacency.indptr[closest_neighbor]:
                                                  adjacency.indptr[closest_neighbor + 1]]
            for v in lex_order[pos_closest + 1:n]:  # If pos_closest = n - 1 it will be empty
                # If the set of other neighbors of vertex (excluding the closest itself) is not a subset of the set
                # of neighbors of the closest, the graph is not chordal.
                if v in neighbors and v not in closest_neighbors:
                    return False
                else:
                    continue

    return True
