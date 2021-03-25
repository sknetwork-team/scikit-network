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


def lexicographic_naive(adjacency: Union[sparse.csr_matrix, np.ndarray]) -> list:
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
    alpha = [-1 for _ in range(n)]

    for i in range(n - 1, -1, -1):
        unnumbered = [v for v in range(n) if alpha[v] < 0]
        # Peut être moyen de mieux mettre à jour ceux avec les plus grands labels?
        # We destroy already used labels later on to guarantee this is safe
        try:
            biggest_label_vertex = np.argmax(labels)
        # If we can't use argmax, it means all labels are empty, in this case just take the first unnumbered vertex
        # showing up.
        except ValueError:
            for j in range(n):
                if alpha[j] < 0:
                    biggest_label_vertex = j
                    break
                # There will always be one because of the for.

        alpha[biggest_label_vertex] = i
        labels[biggest_label_vertex] = []
        # Adding i to the labels of unnumbered adjacent vertices.
        for j in adjacency.indices[adjacency.indptr[biggest_label_vertex]:adjacency.indptr[biggest_label_vertex + 1]]:
            if alpha[j] < 0:
                labels[j].append(str(i))

    lex_order = [0 for _ in range(n)]
    for i in range(n):
        lex_order[alpha[i]] = i

    return lex_order


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
        position[biggest_label_vertex] = i
        print(labels, biggest_label_vertex)
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
    lex_order = lexicographic_naive(adjacency)

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


def fill(adjacency: Union[sparse.csr_matrix, np.ndarray]) -> (list, list):
    alpha = lexicographic_breadth_first_search(adjacency)
    n = adjacency.indptr.shape[0] - 1

    # Initialise test.
    test = [False for _ in range(n)]

    # Storing adjacency lists.
    adjacencies = []
    for i in range(n):
        # TODO ugly and just a fill in to copy the np array to a list and append / remove on it. To be modified.
        adjacencies.append([u for u in adjacency.indices[adjacency.indptr[i]: adjacency.indptr[i + 1]]])

    # alpha_inv stores the position of the vertices in the elimination oder (which is alpha).
    alpha_inv = [0 for _ in range(n)]
    for i in range(n):
        alpha_inv[alpha[i]] = i

    # m is the result.
    m = [-1 for _ in range(n)]

    # Main loop
    for i in range(n - 1):
        k = n - 1
        vertex = alpha[i]

        # Eliminating duplicates in A(vertex)
        for w in adjacencies[vertex]:
            if test[alpha_inv[w]]:
                adjacencies[vertex].remove(w)

            else:
                test[alpha_inv[w]] = True
                k = min(k, alpha_inv[w])

        m[vertex] = alpha[k]
        # Adding required fill in edges and resetting test

        for w in adjacencies[vertex]:
            test[alpha_inv[w]] = False
            if w != m[vertex]:
                adjacencies[m[vertex]].append(w)

    return m, adjacencies


"""
we must check in adjacencies at the end are different than those at the beginning.
"""
