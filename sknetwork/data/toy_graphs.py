#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 29, 2018
@author: Quentin Lutz <qlutz@enst.fr>
@author: Nathan de Lara <nathan.delara@polytechnique.org>
@author: Thomas Bonald <tbonald@enst.fr>
"""
from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.data.base import Dataset


def house(metadata: bool = False) -> Union[sparse.csr_matrix, Dataset]:
    """House graph.

    * Undirected graph
    * 5 nodes, 6 edges

    Parameters
    ----------
    metadata :
        If ``True``, return a `Dataset` object with metadata.

    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Dataset]
        Adjacency matrix or graph with metadata (positions).

    Example
    -------
    >>> from sknetwork.data import house
    >>> adjacency = house()
    >>> adjacency.shape
    (5, 5)

    """
    row = np.array([0, 0, 1, 1, 2, 3])
    col = np.array([1, 4, 2, 4, 3, 4])
    adjacency = sparse.csr_matrix((np.ones(len(row), dtype=int), (row, col)), shape=(5, 5))
    adjacency = (adjacency + adjacency.T).astype(bool)

    if metadata:
        x = np.array([0, -1, -1, 1, 1])
        y = np.array([2, 1, -1, -1, 1])
        graph = Dataset()
        graph.adjacency = adjacency
        graph.position = np.vstack((x, y)).T
        graph.name = 'house'
        return graph
    else:
        return adjacency


def bow_tie(metadata: bool = False) -> Union[sparse.csr_matrix, Dataset]:
    """Bow tie graph.

    * Undirected graph
    * 5 nodes, 6 edges

    Parameters
    ----------
    metadata :
        If ``True``, return a `Dataset` object with metadata.

    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Dataset]
        Adjacency matrix or graph with metadata (positions).

    Example
    -------
    >>> from sknetwork.data import bow_tie
    >>> adjacency = bow_tie()
    >>> adjacency.shape
    (5, 5)
    """
    row = np.array([0, 0, 0, 0, 1, 3])
    col = np.array([1, 2, 3, 4, 2, 4])
    adjacency = sparse.csr_matrix((np.ones(len(row), dtype=int), (row, col)), shape=(5, 5))
    adjacency = (adjacency + adjacency.T).astype(bool)

    if metadata:
        x = np.array([0, -1, 1, -1, 1])
        y = np.array([0, 1, 1, -1, -1])
        graph = Dataset()
        graph.adjacency = adjacency
        graph.position = np.vstack((x, y)).T
        graph.name = 'bow_tie'
        return graph
    else:
        return adjacency


def karate_club(metadata: bool = False) -> Union[sparse.csr_matrix, Dataset]:
    """Karate club graph.

    * Undirected graph
    * 34 nodes, 78 edges
    * 2 labels

    Parameters
    ----------
    metadata :
        If ``True``, return a `Dataset` object with metadata.

    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Dataset]
        Adjacency matrix or graph with metadata (labels, positions).

    Example
    -------
    >>> from sknetwork.data import karate_club
    >>> adjacency = karate_club()
    >>> adjacency.shape
    (34, 34)

    References
    ----------
    Zachary's karate club graph
    https://en.wikipedia.org/wiki/Zachary%27s_karate_club
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
    adjacency = sparse.csr_matrix((np.ones(len(row), dtype=bool), (row, col)), shape=(34, 34))
    adjacency = sparse.csr_matrix(adjacency + adjacency.T, dtype=bool)

    if metadata:
        labels = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        x = np.array(
            [0.04,  0.24,  0.01,  0.13,  0.02, -0.08,  0.04,  0.21,  0.08, -0.11, -0.13, -0.28,  0.2,  0.08,
             0.23,  0.06, -0.06,  0.32, 0.15,  0.19,  0.27,  0.39, -0.04, -0.26, -0.51, -0.49, -0.19, -0.28,
             -0.11, -0.17,  0.22, -0.21,  0.03, 0])
        y = np.array(
            [-0.33, -0.15, -0.01, -0.28, -0.64, -0.75, -0.76, -0.25,  0.09, 0.23, -0.62, -0.4, -0.53, -0.07,
             0.55,  0.64, -1., -0.42, 0.6, -0.01,  0.45, -0.34,  0.61,  0.41,  0.14,  0.28,  0.68, 0.21,
             0.12,  0.54,  0.19,  0.09,  0.38,  0.33])
        graph = Dataset()
        graph.adjacency = adjacency
        graph.labels = labels
        graph.position = np.vstack((x, y)).T
        graph.name = 'karate_club'
        return graph
    else:
        return adjacency


def miserables(metadata: bool = False) -> Union[sparse.csr_matrix, Dataset]:
    """Co-occurrence graph of the characters in the novel Les miserables by Victor Hugo.

    * Undirected graph
    * 77 nodes, 508 edges
    * Names of characters

    Parameters
    ----------
    metadata :
        If ``True``, return a `Dataset` object with metadata.

    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Dataset]
        Adjacency matrix or graph with metadata (names, positions).

    Example
    -------
    >>> from sknetwork.data import miserables
    >>> adjacency = miserables()
    >>> adjacency.shape
    (77, 77)
    """
    row = np.array(
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  2,  3, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
         11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12,
         16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 19, 19,
         20, 20, 20, 21, 21, 22, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25,
         25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27,
         27, 27, 27, 27, 27, 28, 28, 29, 29, 29, 29, 29, 30, 34, 34, 34, 34, 35, 35, 35, 36, 36, 37, 39,
         39, 41, 41, 41, 41, 41, 41, 41, 41, 41, 46, 47, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
         48, 48, 48, 48, 48, 48, 49, 49, 49, 49, 49, 51, 51, 51, 51, 54, 55, 55, 55, 55, 55, 55, 55, 55,
         55, 57, 57, 57, 57, 57, 57, 57, 57, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 59, 59, 59, 59, 59,
         59, 59, 60, 60, 60, 60, 60, 60, 61, 61, 61, 61, 61, 62, 62, 62, 62, 62, 63, 63, 63, 63, 64, 64,
         64, 65, 65, 66, 68, 68, 68, 68, 69, 69, 69, 70, 70, 71, 73])
    col = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 3, 11, 11, 11, 12, 13, 14,
         15, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 43,
         44, 48, 49, 51, 55, 58, 64, 68, 69, 70, 71, 72, 23, 17, 18, 19, 20,
         21, 22, 23, 26, 55, 18, 19, 20, 21, 22, 23, 19, 20, 21, 22, 23, 20,
         21, 22, 23, 21, 22, 23, 22, 23, 23, 24, 25, 27, 29, 30, 31, 25, 26,
         27, 41, 42, 50, 68, 69, 70, 26, 27, 39, 40, 41, 42, 48, 55, 68, 69,
         70, 71, 75, 27, 43, 49, 51, 54, 55, 72, 28, 29, 31, 33, 43, 48, 58,
         68, 69, 70, 71, 72, 44, 45, 34, 35, 36, 37, 38, 31, 35, 36, 37, 38,
         36, 37, 38, 37, 38, 38, 52, 55, 42, 55, 57, 62, 68, 69, 70, 71, 75,
         47, 48, 55, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 69, 71, 73,
         74, 75, 76, 50, 51, 54, 55, 56, 52, 53, 54, 55, 55, 56, 57, 58, 59,
         61, 62, 63, 64, 65, 58, 59, 61, 62, 63, 64, 65, 67, 59, 60, 61, 62,
         63, 64, 65, 66, 70, 76, 60, 61, 62, 63, 64, 65, 66, 61, 62, 63, 64,
         65, 66, 62, 63, 64, 65, 66, 63, 64, 65, 66, 76, 64, 65, 66, 76, 65,
         66, 76, 66, 76, 76, 69, 70, 71, 75, 70, 71, 75, 71, 75, 75, 74])
    data = np.array(
        [1, 8, 10, 1, 1, 1, 1, 2, 1, 5, 6, 3, 3, 1, 1, 1, 1,
         1, 9, 7, 12, 31, 17, 8, 2, 3, 1, 2, 3, 3, 2, 2, 2, 3,
         1, 1, 2, 2, 19, 4, 1, 1, 1, 1, 1, 1, 2, 4, 4, 4, 3,
         3, 3, 3, 1, 1, 4, 4, 3, 3, 3, 3, 4, 3, 3, 3, 3, 4,
         3, 3, 3, 5, 4, 4, 4, 4, 4, 2, 1, 5, 1, 1, 2, 13, 4,
         1, 2, 1, 1, 1, 1, 1, 1, 5, 1, 1, 3, 2, 1, 2, 5, 6,
         4, 1, 3, 1, 1, 3, 2, 1, 21, 2, 1, 1, 1, 1, 1, 1, 6,
         1, 2, 1, 1, 1, 3, 2, 2, 2, 1, 1, 1, 2, 3, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 1, 1, 2, 5, 1, 1, 1, 1, 1, 1, 1,
         1, 2, 4, 1, 7, 6, 1, 2, 7, 5, 5, 3, 1, 1, 1, 1, 2,
         2, 1, 1, 1, 9, 1, 12, 1, 1, 1, 2, 6, 1, 1, 1, 7, 5,
         1, 9, 1, 5, 2, 1, 2, 1, 2, 2, 1, 1, 3, 15, 4, 6, 17,
         4, 10, 5, 3, 1, 1, 2, 5, 13, 5, 9, 5, 1, 2, 3, 2, 2,
         2, 1, 6, 3, 6, 5, 1, 6, 12, 5, 2, 1, 4, 5, 1, 1, 7,
         3, 1, 2, 1, 1, 6, 4, 2, 3, 4, 2, 3, 2, 1, 1, 3])
    adjacency = sparse.csr_matrix((data, (row, col)), shape=(77, 77))
    adjacency = adjacency + adjacency.T

    if metadata:
        names = ['Myriel', 'Napoleon', 'Mlle Baptistine', 'Mme Magloire', 'Countess de Lo', 'Geborand',
                 'Champtercier', 'Cravatte', 'Count', 'Old man', 'Labarre', 'Valjean', 'Marguerite', 'Mme Der',
                 'Isabeau', 'Gervais', 'Tholomyes', 'Listolier', 'Fameuil', 'Blacheville', 'Favourite', 'Dahlia',
                 'Zephine', 'Fantine', 'Mme Thenardier', 'Thenardier', 'Cosette', 'Javert', 'Fauchelevent',
                 'Bamatabois', 'Perpetue', 'Simplice', 'Scaufflaire', 'Woman1', 'Judge', 'Champmathieu', 'Brevet',
                 'Chenildieu', 'Cochepaille', 'Pontmercy', 'Boulatruelle', 'Eponine', 'Anzelma', 'Woman2',
                 'Mother Innocent', 'Gribier', 'Jondrette', 'Mme Burgon', 'Gavroche', 'Gillenormand', 'Magnon',
                 'Mlle Gillenormand', 'Mme Pontmercy', 'Mlle Vaubois', 'Lt Gillenormand', 'Marius', 'Baroness',
                 'Mabeuf', 'Enjolras', 'Combeferre', 'Prouvaire', 'Feuilly', 'Courfeyrac', 'Bahorel', 'Bossuet',
                 'Joly', 'Grantaire', 'MotherPlutarch', 'Gueulemer', 'Babet', 'Claquesous', 'Montparnasse',
                 'Toussaint', 'Child1', 'Child2', 'Brujon', 'Mme Hucheloup']
        x = np.array(
            [0.53,  0.98,  0.41,  0.4,  1.,  0.92,  0.84,  0.74,  0.78, 1.,  0.51,  0.09, -0.,  0.29,  0.37,
             0.41, -0.35, -0.46, -0.42, -0.46, -0.41, -0.37, -0.36, -0.2, -0.06, -0.04, -0.01, -0.02,  0.33,
             0.17, -0.29, -0.1,  0.58,  0.29,  0.29,  0.26, 0.29,  0.37,  0.35,  0.04, -0.01, -0.18, -0.09,
             0.2,  0.51, 0.7, -0.95, -0.7, -0.37, -0.08, -0.18, -0.05,  0.04, -0.12, -0.06, -0.13, -0.24, -0.48,
             -0.25, -0.33, -0.43, -0.39, -0.33, -0.42, -0.31, -0.38, -0.48, -0.74, -0.08, -0.1, -0.02, -0.1,
             0.14, -0.76, -0.75, -0.18, -0.58])
        y = np.array(
            [-0.23, -0.42, -0.14, -0.18, -0.31, -0.52, -0.6, -0.65, -0.38, -0.19,  0.39,  0.03,  0.44, -0.44,
             0.51, -0.36,  0.27,  0.37, 0.4,  0.32,  0.32,  0.36,  0.4,  0.2,  0.07,  0.14, -0.05, 0.06,  0.06,
             0.24, -0.26, -0.1,  0.24, -0.04,  0.17,  0.23, 0.31,  0.21,  0.27, -0.36,  0.69,  0.11,  0.38, -0.09,
             0.05, 0.12,  0.82,  0.44,  0.06, -0.2, -0.4, -0.28, -0.68, -0.79, -0.4, -0.07, -0.51, -0.17, -0.03,
             -0.09, -0.14, -0.04, -0.04, -0.07, -0.06, -0.11, -0.06, -0.35,  0.24,  0.19,  0.22,  0.29, -0.2,
             0.06,  0.14,  0.3, -0.1])
        graph = Dataset()
        graph.adjacency = adjacency
        graph.names = np.array(names)
        graph.position = np.vstack((x, y)).T
        graph.name = 'miserables'
        return graph
    else:
        return adjacency


def painters(metadata: bool = False) -> Union[sparse.csr_matrix, Dataset]:
    """Graph of links between some famous painters on Wikipedia.

    * Directed graph
    * 14 nodes, 50 edges
    * Names of painters

    Parameters
    ----------
    metadata :
        If ``True``, return a `Dataset` object with metadata.

    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Dataset]
        Adjacency matrix or graph with metadata (names, positions).

    Example
    -------
    >>> from sknetwork.data import painters
    >>> adjacency = painters()
    >>> adjacency.shape
    (14, 14)
    """
    row = np.array(
        [0, 0, 1, 1, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5,
         6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9,
         10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13])
    col = np.array(
        [3, 10, 3, 12, 9, 0, 1, 7, 11, 12, 2, 5, 9, 2, 4, 8, 9,
         0, 13, 1, 2, 3, 8, 11, 12, 0, 1, 4, 5, 7, 10, 11, 2, 4,
         0, 3, 8, 11, 12, 0, 1, 3, 10, 12, 1, 3, 4, 7, 6, 8])
    adjacency = sparse.csr_matrix((np.ones(len(row), dtype=bool), (row, col)), shape=(14, 14))

    if metadata:
        names = np.array(
            ['Pablo Picasso', 'Claude Monet', 'Michel Angelo', 'Edouard Manet', 'Peter Paul Rubens', 'Rembrandt',
             'Gustav Klimt', 'Edgar Degas', 'Vincent van Gogh', 'Leonardo da Vinci', 'Henri Matisse', 'Paul Cezanne',
             'Pierre-Auguste Renoir', 'Egon Schiele'])
        x = np.array(
            [0.24, -0.47, -0.3, -0.31, -0.08, 0.12, 0.78, -0.36, 0.11,
             -0.06, -0.02, -0.12, -0.24, 0.73])
        y = np.array(
            [0.53, 0.19, -0.71, 0.44, -0.48, -0.65, 0.69, -0.11, 0.01,
             -1., 0.49, 0.28, 0.06, 0.27])
        graph = Dataset()
        graph.adjacency = adjacency
        graph.names = names
        graph.position = np.stack((x, y)).T
        graph.name = 'painters'
        return graph
    else:
        return adjacency


def hourglass(metadata: bool = False) -> Union[sparse.csr_matrix, Dataset]:
    """Hourglass graph.

    * Bipartite graph
    * 4 nodes, 4 edges

    Returns
    -------
    biadjacency or graph : Union[sparse.csr_matrix, Dataset]
        Biadjacency matrix or graph.

    Example
    -------
    >>> from sknetwork.data import hourglass
    >>> biadjacency = hourglass()
    >>> biadjacency.shape
    (2, 2)
    """
    biadjacency = sparse.csr_matrix(np.ones((2, 2), dtype=bool))
    if metadata:
        graph = Dataset()
        graph.biadjacency = biadjacency
        return graph
    else:
        return biadjacency


def star_wars(metadata: bool = False) -> Union[sparse.csr_matrix, Dataset]:
    """Bipartite graph connecting some Star Wars villains to the movies in which they appear.

    * Bipartite graph
    * 7 nodes (4 villains, 3 movies), 8 edges
    * Names of villains and movies

    Parameters
    ----------
    metadata :
        If ``True``, return a `Dataset` object with metadata.

    Returns
    -------
    biadjacency or graph : Union[sparse.csr_matrix, Dataset]
        Biadjacency matrix or graph with metadata (names).

    Example
    -------
    >>> from sknetwork.data import star_wars
    >>> biadjacency = star_wars()
    >>> biadjacency.shape
    (4, 3)
   """
    row = np.array([0, 0, 1, 2, 2, 2, 3, 3])
    col = np.array([0, 2, 0, 0, 1, 2, 1, 2])
    biadjacency = sparse.csr_matrix((np.ones(len(row), dtype=bool), (row, col)), shape=(4, 3))

    if metadata:
        villains = np.array(['Jabba', 'Greedo', 'Vader', 'Boba'])
        movies = np.array(['A New Hope', 'The Empire Strikes Back', 'Return Of The Jedi'])
        graph = Dataset()
        graph.biadjacency = biadjacency
        graph.names = villains
        graph.names_row = villains
        graph.names_col = movies
        graph.name = 'star_wars'
        return graph
    else:
        return biadjacency


def movie_actor(metadata: bool = False) -> Union[sparse.csr_matrix, Dataset]:
    """Bipartite graph connecting movies to some actors starring in them.

    * Bipartite graph
    * 32 nodes (15 movies, 17 actors), 43 edges
    * Names of movies (rows) and actors (columns)

    Parameters
    ----------
    metadata :
        If ``True``, return a `Dataset` object with metadata.

    Returns
    -------
    biadjacency or dataset : Union[sparse.csr_matrix, Dataset]
        Biadjacency matrix or dataset with metadata (names of movies and actors).

    Example
    -------
    >>> from sknetwork.data import movie_actor
    >>> biadjacency = movie_actor()
    >>> biadjacency.shape
    (15, 17)
    """
    row = np.array(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6,
         6, 6, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11,
         12, 12, 12, 13, 13, 14, 14])
    col = np.array(
        [0, 1, 2, 1, 2, 3, 3, 4, 5, 8, 4, 6, 0, 6, 4, 7, 4,
         7, 8, 3, 8, 9, 10, 11, 12, 15, 0, 11, 12, 9, 10, 13, 16, 5, 9, 13,
         1, 9, 15, 12, 14, 14, 16])
    biadjacency = sparse.csr_matrix((np.ones(len(row), dtype=bool), (row, col)), shape=(15, 17))

    if metadata:
        movies = np.array(
            ['Inception', 'The Dark Knight Rises', 'The Big Short', 'Drive', 'The Great Gatsby', 'La La Land',
             'Crazy Stupid Love', 'Vice', 'The Grand Budapest Hotel', 'Aviator', '007 Spectre', 'Inglourious Basterds',
             'Midnight In Paris', 'Murder on the Orient Express', "Pirates of the Caribbean: At World's End"])
        actors = np.array(
            ['Leonardo DiCaprio', 'Marion Cotillard', 'Joseph Gordon Lewitt', 'Christian Bale', 'Ryan Gosling',
             'Brad Pitt', 'Carey Mulligan', 'Emma Stone', 'Steve Carell', 'Lea Seydoux', 'Ralph Fiennes', 'Jude Law',
             'Willem Dafoe', 'Christophe Waltz', 'Johnny Depp', 'Owen Wilson', 'Naomie Harris'])
        dataset = Dataset()
        dataset.biadjacency = biadjacency
        dataset.names = movies
        dataset.names_row = movies
        dataset.names_col = actors
        dataset.name = 'movie_actor'
        return dataset
    else:
        return biadjacency


def art_philo_science(metadata: bool = False) -> Union[sparse.csr_matrix, Dataset]:
    """Wikipedia links between 30 articles (10 artists, 10 philosophers, 10 scientists).

    * Directed graph
    * 30 nodes, 240 edges
    * Names of articles

    Metadata includes the occurence of 11 words in the abstract of these articles.

    Parameters
    ----------
    metadata :
        If ``True``, return a `Dataset` object with metadata.

    Returns
    -------
    adjacency or graph : Union[sparse.csr_matrix, Dataset]
        Adjacency matrix or graph with metadata (names, positions, labels, names_labels,
        biadjacency, names_col).

    Example
    -------
    >>> from sknetwork.data import art_philo_science
    >>> adjacency = art_philo_science()
    >>> adjacency.shape
    (30, 30)
    """
    row = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6,
         6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8,
         8, 8, 8, 8, 8, 9, 9, 10, 10, 10, 10, 11, 11, 11, 12, 12, 12,
         12, 12, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17,
         17, 17, 17, 18, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
         20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
         21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23,
         23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24,
         24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 25,
         25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
         26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
         28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29,
         29, 29])
    col = np.array(
        [1, 5, 6, 7, 20, 21, 23, 24, 25, 26, 27, 28, 0, 5, 7, 9, 12,
         16, 20, 21, 24, 25, 26, 27, 28, 3, 20, 0, 2, 26, 7, 20, 21, 29,
         0, 1, 6, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 0, 1,
         5, 7, 8, 24, 26, 27, 0, 1, 6, 20, 21, 24, 25, 26, 27, 28, 0,
         5, 6, 7, 9, 27, 1, 25, 0, 18, 19, 20, 12, 16, 17, 11, 12, 14,
         16, 17, 14, 15, 25, 11, 12, 16, 17, 19, 13, 14, 19, 12, 14, 16, 11,
         12, 14, 16, 10, 14, 17, 18, 0, 1, 2, 3, 4, 5, 7, 18, 21, 22,
         23, 24, 25, 26, 27, 28, 29, 0, 1, 4, 5, 7, 18, 20, 22, 23, 24,
         25, 26, 27, 28, 29, 5, 20, 21, 23, 24, 25, 26, 27, 28, 29, 5, 20,
         21, 22, 23, 24, 25, 26, 27, 28, 29, 0, 1, 2, 5, 6, 7, 10, 20,
         21, 22, 23, 25, 26, 27, 28, 29, 0, 1, 5, 7, 9, 13, 20, 21, 22,
         23, 24, 26, 27, 28, 29, 0, 1, 5, 6, 7, 20, 21, 22, 23, 24, 25,
         27, 28, 29, 0, 1, 3, 5, 6, 7, 20, 21, 22, 23, 24, 25, 26, 28,
         0, 1, 5, 7, 9, 20, 21, 22, 23, 24, 25, 26, 27, 29, 4, 18, 20,
         21, 22])
    adjacency = sparse.csr_matrix((np.ones(len(row), dtype=bool), (row, col)), shape=(30, 30))

    if metadata:
        names = np.array(
            ['Isaac Newton',
             'Albert Einstein',
             'Carl Linnaeus',
             'Charles Darwin',
             'Ptolemy',
             'Gottfried Wilhelm Leibniz',
             'Carl Friedrich Gauss',
             'Galileo Galilei',
             'Leonhard Euler',
             'John von Neumann',
             'Leonardo da Vinci',
             'Richard Wagner',
             'Ludwig van Beethoven',
             'Bob Dylan',
             'Igor Stravinsky',
             'The Beatles',
             'Wolfgang Amadeus Mozart',
             'Richard Strauss',
             'Raphael',
             'Pablo Picasso',
             'Aristotle',
             'Plato',
             'Augustine of Hippo',
             'Thomas Aquinas',
             'Immanuel Kant',
             'Bertrand Russell',
             'David Hume',
             'René Descartes',
             'John Stuart Mill',
             'Socrates'])
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        names_labels = np.array(['science', 'arts', 'philosophy'])
        position = np.array(
            [[0.25, 0.57],
             [0.28, 0.37],
             [-0.08, 0.82],
             [0.08, 0.87],
             [-0.08, 0.49],
             [0.31, 0.58],
             [0.41, 0.66],
             [0.31, 0.51],
             [0.53, 0.65],
             [0.48, 0.42],
             [-0.02, 0.29],
             [0.25, -0.44],
             [0.23, -0.22],
             [0.02, 0.03],
             [0.07, -0.26],
             [-0.16, -0.23],
             [0.28, -0.15],
             [0.12, -0.38],
             [-0.14, 0.29],
             [-0.09, -0.1],
             [0.09, 0.57],
             [0.12, 0.5],
             [0.16, 0.7],
             [0.12, 0.66],
             [0.17, 0.57],
             [0.2, 0.44],
             [0.22, 0.64],
             [0.28, 0.65],
             [0.23, 0.51],
             [-0., 0.57]])
        row = np.array(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3,
             3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 6, 7, 7, 7, 7, 8, 8,
             8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 12, 12, 12,
             13, 13, 14, 15, 15, 16, 16, 17, 17, 18, 19, 19, 19, 19, 20, 20, 20,
             20, 20, 20, 21, 21, 21, 21, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24,
             25, 25, 25, 26, 26, 27, 27, 27, 27, 28, 28, 28, 29, 29, 29, 29])
        col = np.array(
            [1, 2, 4, 5, 10, 2, 3, 4, 5, 8, 1, 4, 8, 10, 1, 2, 4,
             5, 8, 10, 2, 9, 2, 5, 7, 8, 9, 5, 1, 4, 5, 8, 1, 2,
             3, 5, 8, 2, 4, 5, 0, 4, 5, 6, 8, 5, 8, 9, 1, 4, 6,
             4, 10, 9, 4, 9, 4, 6, 4, 8, 9, 0, 2, 8, 9, 1, 2, 3,
             7, 8, 9, 2, 3, 4, 5, 2, 5, 1, 2, 3, 4, 8, 2, 4, 8,
             2, 7, 9, 2, 4, 3, 4, 8, 9, 2, 9, 10, 5, 6, 8, 10])
        data = np.array(
            [2, 2, 3, 1, 1, 15, 1, 2, 2, 2, 1, 1, 3, 1, 1, 7, 2,
             1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 1, 1, 1, 1, 2, 3, 3,
             1, 1, 1, 6, 1, 1, 1, 3, 1, 1, 1, 1, 1, 2, 1, 1, 2,
             4, 1, 2, 2, 1, 3, 3, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1,
             4, 1, 2, 1, 1, 1, 1, 1, 1, 3, 1, 2, 2, 2, 3, 1, 2,
             2, 5, 1, 1, 1, 3, 1, 5, 5, 3, 1, 1, 1, 1, 1, 2])
        biadjacency = sparse.csr_matrix((data, (row, col)), shape=(30, 11))
        words = np.array(
            ['contribution', 'theory', 'invention', 'time', 'modern',
             'century', 'study', 'logic', 'school', 'author', 'compose'])
        graph = Dataset()
        graph.adjacency = adjacency
        graph.names = names
        graph.position = position
        graph.labels = labels
        graph.names_labels = names_labels
        graph.biadjacency = biadjacency
        graph.names_col = words
        graph.name = 'art-philo-science'
        return graph
    else:
        return adjacency
