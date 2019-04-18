#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 29, 2018
@author: Quentin Lutz <qlutz@enst.fr>
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <tbonald@enst.ft>
"""

import numpy as np
from scipy import sparse


def house_graph():
    """
    House graph

    5 nodes, 6 edges

    Returns
    -------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph.
    """
    row = np.array([0, 0, 1, 1, 2, 3])
    col = np.array([1, 4, 2, 4, 3, 4])
    adjacency = sparse.csr_matrix((np.ones(len(row), dtype=int), (row, col)), shape=(5, 5))
    return adjacency + adjacency.T


def karate_club_graph():
    """
    Zachary's Karate Club Graph

    Data file from: http://vlado.fmf.uni-lj.si/pub/networks/data/Ucinet/UciData.htm

    34 nodes, 78 edges

    Returns
    -------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph.
    """
    row = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
         3, 4, 4, 5, 5, 5, 6, 8, 8, 8, 9, 13, 14, 14, 15, 15, 18,
         18, 19, 20, 20, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 25, 26, 26,
         27, 28, 28, 29, 29, 30, 30, 31, 31, 32])
    col = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 2,
         3, 7, 13, 17, 19, 21, 30, 3, 7, 8, 9, 13, 27, 28, 32, 7, 12,
         13, 6, 10, 6, 10, 16, 16, 30, 32, 33, 33, 33, 32, 33, 32, 33, 32,
         33, 33, 32, 33, 32, 33, 25, 27, 29, 32, 33, 25, 27, 31, 31, 29, 33,
         33, 31, 33, 32, 33, 32, 33, 32, 33, 33])
    adjacency = sparse.csr_matrix((np.ones(len(row), dtype=int), (row, col)), shape=(34, 34))
    return adjacency + adjacency.T


def rock_paper_scissors_graph():
    """A toy directed cycle graph from Rock Paper Scissors victory rule.

    3 nodes, 3 edges

    Returns
    -------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph.

    """

    return sparse.csr_matrix(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))


def star_wars_villains_graph(return_labels=False):
    """
    Bipartite graph connecting some Star Wars villains to the movies in which they appear.\n
    7 nodes (4 villains, 3 movies), 8 edges

    Parameters
    ----------
    return_labels: bool
        whether to return the labels of the nodes as dictionaries.

    Returns
    -------
    biadjacency: sparse.csr_matrix
        Biadjacency matrix of the graph.
    """
    row = np.array([0, 0, 1, 2, 2, 2, 3, 3])
    col = np.array([0, 2, 0, 0, 1, 2, 1, 2])
    biadjacency = sparse.csr_matrix((np.ones(len(row), dtype=int), (row, col)))

    if return_labels:
        row_labels = {0: 'Jabba', 1: 'Greedo', 2: 'Vador', 3: 'Boba'}
        col_labels = {0: 'A_New_Hope', 1: 'The_Empire_Strikes_Back', 2: 'Return_Of_The_Jedi'}
        return biadjacency, row_labels, col_labels
    else:
        return biadjacency


def movie_actor_graph(return_labels=False):
    """
    Bipartite graph connecting movies to some actors starring in them.\n
    31 nodes (15 movies, 16 actors), 41 edges

    Parameters
    ----------
    return_labels: bool
        whether to return the labels of the nodes as dictionaries.

    Returns
    -------
    biadjacency: sparse.csr_matrix
        Biadjacency matrix of the graph.
    """
    edges = {
        0: [0, 1, 2],
        1: [1, 2, 3],
        2: [3, 4, 5, 8],
        3: [4, 6],
        4: [0, 6],
        5: [4, 7],
        6: [4, 7, 8],
        7: [3, 8],
        8: [9, 10, 11, 12, 15],
        9: [0, 11, 12],
        10: [9, 10],
        11: [5, 9, 13],
        12: [1, 9, 15],
        13: [12, 14],
        14: [11, 14]
    }
    row, col = [], []
    for key, item in edges.items():
        row += [key] * len(item)
        col += item
    biadjacency = sparse.csr_matrix((np.ones(len(row), dtype=int), (row, col)))

    if return_labels:
        row_labels = {
            0: 'Inception',
            1: 'The Dark Knight Rises',
            2: 'The Big Short',
            3: 'Drive',
            4: 'The Great Gatsby',
            5: 'La La Land',
            6: 'Crazy Stupid Love',
            7: 'Vice',
            8: 'The Grand Budapest Hotel',
            9: 'Aviator',
            10: '007 Spectre',
            11: 'Inglourious Basterds',
            12: 'Midnight In Paris',
            13: 'Murder on the Orient Express',
            14: 'Fantastic Beasts 2'
        }
        col_labels = {
            0: 'Leonardo DiCaprio',
            1: 'Marion Cotillard',
            2: 'Joseph Gordon Lewitt',
            3: 'Christian Bale',
            4: 'Ryan Gosling',
            5: 'Brad Pitt',
            6: 'Carey Mulligan',
            7: 'Emma Stone',
            8: 'Steve Carell',
            9: 'Lea Seydoux',
            10: 'Ralph Fiennes',
            11: 'Jude Law',
            12: 'Willem Dafoe',
            13: 'Christophe Waltz',
            14: 'Johnny Depp',
            15: 'Owen Wilson'
        }
        return biadjacency, row_labels, col_labels
    else:
        return biadjacency
