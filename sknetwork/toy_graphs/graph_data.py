#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 29, 2018
@author: Quentin Lutz <qlutz@enst.fr>
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <tbonald@enst.fr>
"""

import numpy as np
from scipy import sparse


def miserables_graph(return_labels=False):
    """
    Co-occurrence graph of the characters in Les Miserables (by Victor Hugo).

    77 nodes, 508 edges

    Returns
    -------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph.
    labels: dict, optional
        The names of the characters corresponding to the nodes in the graph.
    """
    indptr = np.array([0,  10,  11,  14,  17,  18,  19,  20,  21,  22,  23,  24,  60,
                       62,  63,  64,  65,  74,  81,  88,  95, 102, 109, 116, 131, 142,
                       158, 169, 186, 190, 198, 200, 204, 205, 207, 213, 219, 225, 231,
                       237, 240, 241, 252, 255, 258, 260, 261, 262, 264, 286, 293, 295,
                       302, 304, 305, 309, 328, 330, 341, 356, 367, 376, 387, 400, 412,
                       425, 437, 447, 448, 458, 468, 478, 487, 490, 492, 494, 501, 508])
    indices = np.array([1,  2,  3,  4,  5,  6,  7,  8,  9, 11,  0,  0,  3, 11,  0,  2, 11,
                       0,  0,  0,  0,  0,  0, 11,  0,  2,  3, 10, 12, 13, 14, 15, 23, 24,
                       25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 43, 44, 48, 49,
                       51, 55, 58, 64, 68, 69, 70, 71, 72, 11, 23, 11, 11, 11, 17, 18, 19,
                       20, 21, 22, 23, 26, 55, 16, 18, 19, 20, 21, 22, 23, 16, 17, 19, 20,
                       21, 22, 23, 16, 17, 18, 20, 21, 22, 23, 16, 17, 18, 19, 21, 22, 23,
                       16, 17, 18, 19, 20, 22, 23, 16, 17, 18, 19, 20, 21, 23, 11, 12, 16,
                       17, 18, 19, 20, 21, 22, 24, 25, 27, 29, 30, 31, 11, 23, 25, 26, 27,
                       41, 42, 50, 68, 69, 70, 11, 23, 24, 26, 27, 39, 40, 41, 42, 48, 55,
                       68, 69, 70, 71, 75, 11, 16, 24, 25, 27, 43, 49, 51, 54, 55, 72, 11,
                       23, 24, 25, 26, 28, 29, 31, 33, 43, 48, 58, 68, 69, 70, 71, 72, 11,
                       27, 44, 45, 11, 23, 27, 34, 35, 36, 37, 38, 23, 31, 11, 23, 27, 30,
                       11, 11, 27, 11, 29, 35, 36, 37, 38, 11, 29, 34, 36, 37, 38, 11, 29,
                       34, 35, 37, 38, 11, 29, 34, 35, 36, 38, 11, 29, 34, 35, 36, 37, 25,
                       52, 55, 25, 24, 25, 42, 55, 57, 62, 68, 69, 70, 71, 75, 24, 25, 41,
                       11, 26, 27, 11, 28, 28, 47, 46, 48, 11, 25, 27, 47, 55, 57, 58, 59,
                       60, 61, 62, 63, 64, 65, 66, 68, 69, 71, 73, 74, 75, 76, 11, 26, 50,
                       51, 54, 55, 56, 24, 49, 11, 26, 49, 52, 53, 54, 55, 39, 51, 51, 26,
                       49, 51, 55, 11, 16, 25, 26, 39, 41, 48, 49, 51, 54, 56, 57, 58, 59,
                       61, 62, 63, 64, 65, 49, 55, 41, 48, 55, 58, 59, 61, 62, 63, 64, 65,
                       67, 11, 27, 48, 55, 57, 59, 60, 61, 62, 63, 64, 65, 66, 70, 76, 48,
                       55, 57, 58, 60, 61, 62, 63, 64, 65, 66, 48, 58, 59, 61, 62, 63, 64,
                       65, 66, 48, 55, 57, 58, 59, 60, 62, 63, 64, 65, 66, 41, 48, 55, 57,
                       58, 59, 60, 61, 63, 64, 65, 66, 76, 48, 55, 57, 58, 59, 60, 61, 62,
                       64, 65, 66, 76, 11, 48, 55, 57, 58, 59, 60, 61, 62, 63, 65, 66, 76,
                       48, 55, 57, 58, 59, 60, 61, 62, 63, 64, 66, 76, 48, 58, 59, 60, 61,
                       62, 63, 64, 65, 76, 57, 11, 24, 25, 27, 41, 48, 69, 70, 71, 75, 11,
                       24, 25, 27, 41, 48, 68, 70, 71, 75, 11, 24, 25, 27, 41, 58, 68, 69,
                       71, 75, 11, 25, 27, 41, 48, 68, 69, 70, 75, 11, 26, 27, 48, 74, 48,
                       73, 25, 41, 48, 68, 69, 70, 71, 48, 58, 62, 63, 64, 65, 66])
    data = np.array([1,  8, 10,  1,  1,  1,  1,  2,  1,  5,  1,  8,  6,  3, 10,  6,  3,
                    1,  1,  1,  1,  2,  1,  1,  5,  3,  3,  1,  1,  1,  1,  1,  9,  7,
                    12, 31, 17,  8,  2,  3,  1,  2,  3,  3,  2,  2,  2,  3,  1,  1,  2,
                    2, 19,  4,  1,  1,  1,  1,  1,  1,  1,  2,  1,  1,  1,  4,  4,  4,
                    3,  3,  3,  3,  1,  1,  4,  4,  4,  3,  3,  3,  3,  4,  4,  4,  3,
                    3,  3,  3,  4,  4,  4,  4,  3,  3,  3,  3,  3,  3,  4,  5,  4,  4,
                    3,  3,  3,  3,  5,  4,  4,  3,  3,  3,  3,  4,  4,  4,  9,  2,  3,
                    3,  3,  3,  4,  4,  4,  2,  1,  5,  1,  1,  2,  7,  2, 13,  4,  1,
                    2,  1,  1,  1,  1,  1, 12,  1, 13,  1,  5,  1,  1,  3,  2,  1,  2,
                    5,  6,  4,  1,  3, 31,  1,  4,  1,  1,  1,  3,  2,  1, 21,  2, 17,
                    5,  1,  5,  1,  1,  1,  1,  1,  1,  1,  6,  1,  2,  1,  1,  1,  8,
                    1,  3,  2,  2,  1,  1,  2,  2,  1,  1,  1,  1,  2,  3,  2,  1,  2,
                    1,  2,  1,  3,  2,  3,  2,  2,  2,  3,  2,  3,  2,  2,  2,  2,  1,
                    2,  2,  2,  2,  2,  1,  2,  2,  2,  2,  2,  1,  2,  2,  2,  2,  1,
                    1,  1,  1,  2,  3,  2,  5,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,
                    3,  1,  1,  1,  3,  2,  1,  1,  2,  1,  1,  1,  2,  4,  1,  7,  6,
                    1,  2,  7,  5,  5,  3,  1,  1,  1,  1,  2,  2,  1,  1,  2,  3,  1,
                    9,  1, 12,  1,  1,  1,  2,  2,  9,  1,  1,  2,  6,  1,  1,  1,  1,
                    1,  2,  1, 19,  1,  2, 21,  1,  5,  4, 12,  6,  1,  1,  1,  7,  5,
                    1,  9,  1,  5,  2,  1,  1,  1,  1,  1,  1,  2,  1,  2,  2,  1,  1,
                    3,  4,  6,  7,  7,  1, 15,  4,  6, 17,  4, 10,  5,  3,  1,  1,  6,
                    5,  2, 15,  2,  5, 13,  5,  9,  5,  1,  1,  4,  2,  2,  3,  2,  2,
                    2,  1,  2,  1,  1,  6,  5,  2,  6,  3,  6,  5,  1,  1,  7,  9,  2,
                    17, 13,  3,  6,  6, 12,  5,  2,  1,  5,  1,  2,  4,  5,  2,  3,  6,
                    4,  5,  1,  1,  1,  5,  5,  1, 10,  9,  2,  6, 12,  4,  7,  3,  1,
                    3,  2,  1,  5,  5,  2,  5,  5,  5,  7,  2,  1,  1,  3,  1,  1,  1,
                    2,  1,  3,  2,  1,  3,  1,  1,  5,  1,  1,  1,  6,  4,  2,  3,  1,
                    1,  6,  2,  1,  1,  6,  4,  2,  3,  1,  1,  4,  1,  1,  1,  4,  4,
                    2,  1,  1,  1,  1,  1,  1,  2,  2,  2,  1,  1,  2,  1,  2,  3,  2,
                    3,  3,  1,  1,  3,  3,  1,  1,  1,  1,  1,  1,  1,  1,  1])
    adjacency = sparse.csr_matrix((data, indices, indptr))
    if return_labels:
        labels = {0: 'Myriel',
                  1: 'Napoleon',
                  2: 'Mlle Baptistine',
                  3: 'Mme Magloire',
                  4: 'Countess de Lo',
                  5: 'Geborand',
                  6: 'Champtercier',
                  7: 'Cravatte',
                  8: 'Count',
                  9: 'Old man',
                  10: 'Labarre',
                  11: 'Valjean',
                  12: 'Marguerite',
                  13: 'Mme Der',
                  14: 'Isabeau',
                  15: 'Gervais',
                  16: 'Tholomyes',
                  17: 'Listolier',
                  18: 'Fameuil',
                  19: 'Blacheville',
                  20: 'Favourite',
                  21: 'Dahlia',
                  22: 'Zephine',
                  23: 'Fantine',
                  24: 'Mme Thenardier',
                  25: 'Thenardier',
                  26: 'Cosette',
                  27: 'Javert',
                  28: 'Fauchelevent',
                  29: 'Bamatabois',
                  30: 'Perpetue',
                  31: 'Simplice',
                  32: 'Scaufflaire',
                  33: 'Woman1',
                  34: 'Judge',
                  35: 'Champmathieu',
                  36: 'Brevet',
                  37: 'Chenildieu',
                  38: 'Cochepaille',
                  39: 'Pontmercy',
                  40: 'Boulatruelle',
                  41: 'Eponine',
                  42: 'Anzelma',
                  43: 'Woman2',
                  44: 'MotherInnocent',
                  45: 'Gribier',
                  46: 'Jondrette',
                  47: 'Mme Burgon',
                  48: 'Gavroche',
                  49: 'Gillenormand',
                  50: 'Magnon',
                  51: 'Mlle Gillenormand',
                  52: 'Mme Pontmercy',
                  53: 'Mlle Vaubois',
                  54: 'Lt Gillenormand',
                  55: 'Marius',
                  56: 'Baroness',
                  57: 'Mabeuf',
                  58: 'Enjolras',
                  59: 'Combeferre',
                  60: 'Prouvaire',
                  61: 'Feuilly',
                  62: 'Courfeyrac',
                  63: 'Bahorel',
                  64: 'Bossuet',
                  65: 'Joly',
                  66: 'Grantaire',
                  67: 'MotherPlutarch',
                  68: 'Gueulemer',
                  69: 'Babet',
                  70: 'Claquesous',
                  71: 'Montparnasse',
                  72: 'Toussaint',
                  73: 'Child1',
                  74: 'Child2',
                  75: 'Brujon',
                  76: 'Mme Hucheloup'}
        return adjacency, labels
    return adjacency


def bow_tie_graph():
    """
    Bow tie graph

    5 nodes, 6 edges

    Returns
    -------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph.
    """
    row = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    col = np.array([1, 2, 3, 4, 0, 2, 0, 1, 0, 4, 0, 3])
    adjacency = sparse.csr_matrix((np.ones(len(row), dtype=int), (row, col)), shape=(5, 5))
    return adjacency


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
