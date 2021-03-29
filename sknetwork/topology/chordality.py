#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 17, 2020
@author: Alexis Barreaux <alexis.barreaux@telecom-paris.fr>
"""
from typing import Union, List

import numpy as np
import random as rd
from scipy import sparse
from traitlets import Tuple

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


def lexicographic_naive(adjacency: Union[sparse.csr_matrix, np.ndarray]) -> Tuple(List[int], List[int]):
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
    alpha_inv = [-1 for _ in range(n)]

    for i in range(n - 1, -1, -1):
        # Peut être moyen de mieux mettre à jour ceux avec les plus grands labels?
        # We destroy already used labels later on to guarantee this is safe
        try:
            biggest_label_vertex = np.argmax(labels)
        # If we can't use argmax, it means all labels are empty, in this case just take the first unnumbered vertex
        # showing up.
        except ValueError:
            for j in range(n):
                if alpha_inv[j] < 0:
                    biggest_label_vertex = j
                    break
                # There will always be one because of the for.

        alpha_inv[biggest_label_vertex] = i
        labels[biggest_label_vertex] = []
        # Adding i to the labels of unnumbered adjacent vertices.
        for j in adjacency.indices[adjacency.indptr[biggest_label_vertex]:adjacency.indptr[biggest_label_vertex + 1]]:
            if alpha_inv[j] < 0:
                labels[j].append(str(i))

    alpha = [0 for _ in range(n)]
    for i in range(n):
        alpha[alpha_inv[i]] = i

    # Pour moi, lex_order[::-1] est un bfs.
    return alpha, alpha_inv


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


def lexicographic_breadth_first_search_v3(adjacency: Union[sparse.csr_matrix, np.ndarray]) -> Tuple(List[int], List[int]):
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
    alpha_inv = [-1 for _ in range(n)]

    vertices_sets = dict()
    vertices_sets[0] = [-1, -1, set([i for i in range(n)])]
    current_set_positions = [0 for _ in range(n)]
    last_set_id = 0
    max_set_id = 0
    """
    chaque set est une liste comprenant :
        - le nom du set précédent ou -1
        - le nom du set suivant ou -1
        - les noeuds qui ont comme label celui associé au set
        - un identifiant unique (dans un dict par exemple)
    """
    for i in range(n - 1, -1, -1):
        # Getting a vertex with max label.
        while len(vertices_sets[last_set_id][2]) == 0:
            # TODO check why one of this MAJ of last set id is done wrong or why we update previous / next wrongly
            assert(vertices_sets[last_set_id][1] == -1)
            # If the last set is empty (we popped all the vertices)
            # We update the next and previous sets and then delete our empty set. Since it is the last, he has not next
            # set but might have a previous one.
            if vertices_sets[last_set_id][0] == -1:
                # The previous one doesn't exist, in this case we are going to finish
                raise Exception("Sets are empty too early")
                # This shouldn't happen, since we shouldn't have no previous in our last set.
            else:
                # There is a previous set
                vertices_sets[vertices_sets[last_set_id][0]][1] = -1
                # Our previous set is going to have his next set be transformed to -1

            # Update last_set_id
            new_last = vertices_sets[last_set_id][0]
            if last_set_id == 34:
                print("11111111111111111")
            del vertices_sets[last_set_id]
            last_set_id = new_last

        print(vertices_sets)

        cur_vertex = vertices_sets[last_set_id][2].pop()
        # Setting alpha_inv
        alpha_inv[cur_vertex] = i

        new_sets = dict()
        new_last = last_set_id
        for w in adjacency.indices[adjacency.indptr[cur_vertex]:adjacency.indptr[cur_vertex + 1]]:
            # Checking if this vertex has already been numbered or not:
            if alpha_inv[w] >= 0:
                # If it has, don't take it
                continue
            else:
                # We check for all these neighbors where we need to take them from
                try:
                    # If the set is not already added:
                    new_sets[current_set_positions[w]].append(w)

                except KeyError:
                    new_sets[current_set_positions[w]] = [w]

        for k in new_sets:
            for w in new_sets[k]:
                vertices_sets[k][2].remove(w)
                try:
                    # Try to add this neighbor in his new set
                    vertices_sets[max_set_id + 1][2].add(w)
                except KeyError:
                    # Otherwise creates this new set
                    suc_of_prec = vertices_sets[k][1]
                    vertices_sets[max_set_id + 1] = [k, suc_of_prec, set([w])]
                    if suc_of_prec != -1:
                        vertices_sets[suc_of_prec][0] = max_set_id + 1
                    else:
                        new_last = max_set_id + 1
                    vertices_sets[k][1] = max_set_id + 1

                current_set_positions[w] = max_set_id + 1

            max_set_id += 1

            """
            # Remove empty sets here.
            if len(vertices_sets[k][2]) == 0:
                if vertices_sets[k][0] == -1:
                    # The previous one doesn't exist, in this case we are going to continue
                    if vertices_sets[k][1] == -1:
                        pass
                    else:
                        # We must update the previous set of the next set.
                        vertices_sets[vertices_sets[k][1]][0] = -1
                else:
                    # There is a previous set.
                    if vertices_sets[k][1] == -1:
                        # There is no next set
                        vertices_sets[vertices_sets[k][0]][1] = -1
                    else:
                        vertices_sets[vertices_sets[k][0]][1] = vertices_sets[k][1]
                        vertices_sets[vertices_sets[k][1]][0] = vertices_sets[k][0]

                del vertices_sets[k]
            """


        last_set_id = new_last

    alpha = [0 for _ in range(n)]
    for i in range(n):
        alpha[alpha_inv[i]] = i
    return alpha, alpha_inv


def lexicographic_linear(adjacency: Union[sparse.csr_matrix, np.ndarray]) -> Tuple(List[int], List[int]):
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
    alpha = [-1 for _ in range(2*n)]
    alpha_inv = [0 for _ in range(2*n)]

    head = [0 for _ in range(2*n)]
    back = [0 for _ in range(2*n)]
    next = [0 for _ in range(2*n)]
    flag = [0 for _ in range(2*n)]
    cell = [0 for _ in range(2*n)]

    # Init
    head[0], head[1] = 2, 0
    back[0], back[1] = 0, 1
    next[0] = 0
    flag[0], flag[1] = 0, 0
    # c is the number of the first empty cell.
    c = 2

    for i in range(n):
        head[c] = i
        cell[i] = c
        next[c - 1] = c
        flag[c] = 1
        back[c] = c - 1
        c += 1
        alpha_inv[i] = 0

    next[c - 1] = 0
    for i in range(n - 1, -1, -1):
        while next[head[0]] == 0:
            head[0] = head[head[0]]
            back[head[0]] = 1

        # TODO pas sûr ici
        p = next[head[0]]
        next[head[0]] = next[p]
        # TODO bizarre le truc avec des w ici
        alpha[i] = p
        alpha_inv[p] = i
        fixlist = []

        for w in adjacency.indices[adjacency.indptr[p]: adjacency.indptr[p + 1]]:
            if alpha_inv[w] == 0:
                next[back[cell[w]]] = next[cell[w]]

                if next[cell[w]] != 0:
                    back[next[cell[w]]] = back[cell[w]]

                h = back[flag[cell[w]]]

                if flag[h] == 0:
                    print(c, h)
                    head[c] = head[h]
                    head[h] = c
                    back[head[c]] = c
                    back[c] = h
                    flag[c] = 1
                    next[c] = 0
                    fixlist.append(c)
                    h = c
                    c += 1

                next[cell[w]] = next[h]
                if next[h] != 0:
                    back[next[h]] = cell[w]
                flag[cell[w]] = h
                back[cell[w]] = h
                next[h] = cell[w]

            for h in fixlist:
                flag[h] = p

    return alpha, alpha_inv


def fill_naive(adjacency: Union[sparse.csr_matrix, np.ndarray]) -> (list, list):
    alpha, alpha_inv = lexicographic_naive(adjacency)
    n = adjacency.indptr.shape[0] - 1
    m = [0 for _ in range(n)]

    adjacencies = []
    for i in range(n):
        # TODO ugly and just a fill in to copy the np array to a list and append / remove on it. To be modified.
        adjacencies.append([u for u in adjacency.indices[adjacency.indptr[i]: adjacency.indptr[i + 1]]])

    for i in range(n - 1):
        v = alpha[i]
        m[v] = alpha[min([alpha_inv[w] for w in adjacencies[v]])]

        for w in adjacencies[v]:
            if w != m[v]:
                adjacencies[m[v]].append(w)

    return adjacencies


def fill(adjacency: Union[sparse.csr_matrix, np.ndarray]) -> (list, list):
    # TODO utiliser l'objet set de python plutôt que des listes
    alpha, alpha_inv = lexicographic_naive(adjacency)
    n = adjacency.indptr.shape[0] - 1

    # Initialise test.
    test = [False for _ in range(n)]

    # Storing adjacency lists.
    adjacencies = []
    for i in range(n):
        # TODO ugly and just a fill in to copy the np array to a list and append / remove on it. To be modified.
        adjacencies.append([u for u in adjacency.indices[adjacency.indptr[i]: adjacency.indptr[i + 1]]])

    # m as used in the paper.
    m = [-1 for _ in range(n)]

    # Main loop
    for i in range(n - 1):
        k = n - 1
        vertex = alpha[i]

        # Eliminating duplicates in A(vertex)
        for w in adjacencies[vertex]:
            if test[alpha_inv[w]]:
                # TODO this isn't linear, I could store indexes when I have a duplicate and attempt to pop them afterwards
                adjacencies[vertex].remove(w)

            else:
                test[alpha_inv[w]] = True
                k = min(k, alpha_inv[w])

        m[vertex] = alpha[k]
        # Adding required fill in edges and resetting test

        for w in adjacencies[vertex]:
            test[alpha_inv[w]] = False
            if w != m[vertex]:
                # TODO trouver un moyen linéaire en la taille de G de sortir ici si on ajoute un arc pas pré existant.
                adjacencies[m[vertex]].append(w)

    return adjacencies


def fill_2(adjacency: Union[sparse.csr_matrix, np.ndarray]) -> (list, int):
    alpha, alpha_inv = lexicographic_breadth_first_search_v3(adjacency)
    n = adjacency.indptr.shape[0] - 1

    # Initialise test.
    test = [False for _ in range(n)]

    # Storing adjacency lists and calculating initial len.
    init_len = 0
    adjacencies = []
    for i in range(n):
        adjacencies.append(set(adjacency.indices[adjacency.indptr[i]: adjacency.indptr[i + 1]]))
        init_len += len(adjacencies[i])

    # m as used in the paper.
    m = [-1 for _ in range(n)]

    # Main loop
    for i in range(n - 1):
        k = n - 1
        vertex = alpha[i]

        # Eliminating duplicates in A(vertex)
        for w in adjacencies[vertex]:
            if test[alpha_inv[w]]:
                # In a set we don't need to remove anymore.
                continue

            else:
                test[alpha_inv[w]] = True
                k = min(k, alpha_inv[w])

        m[vertex] = alpha[k]
        # Adding required fill in edges and resetting test

        for w in adjacencies[vertex]:
            test[alpha_inv[w]] = False
            if w != m[vertex]:
                # Should be linear too
                adjacencies[m[vertex]].add(w)

    return adjacencies, init_len


def is_chordal_other(adjacency: Union[sparse.csr_matrix, np.ndarray]) -> bool:
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
    adjacencies, init_len = fill_2(adjacency)

    n = adjacency.indptr.shape[0] - 1
    final_len = 0
    for i in range(n):
        final_len += len(adjacencies[i])
    return final_len == init_len
