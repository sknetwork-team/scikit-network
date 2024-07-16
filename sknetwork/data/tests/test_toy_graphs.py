# -*- coding: utf-8 -*-
# tests for toy_graphs.py
"""
@author: Quentin Lutz <qlutz@enst.fr>
@author: Nathan de Lara <nathan.delara@polytechnique.org>
@author: Thomas Bonald <tbonald@enst.fr>
"""
import unittest

from sknetwork.data.toy_graphs import *


class TestToys(unittest.TestCase):

    def test_undirected(self):
        adjacency = house()
        self.assertEqual(adjacency.shape, (5, 5))

        dataset = house(metadata=True)
        self.assertEqual(dataset.position.shape, (5, 2))

        adjacency = bow_tie()
        self.assertEqual(adjacency.shape, (5, 5))

        dataset = bow_tie(metadata=True)
        self.assertEqual(dataset.position.shape, (5, 2))

        dataset = karate_club(True)
        self.assertEqual(dataset.adjacency.shape, (34, 34))
        self.assertEqual(len(dataset.labels), 34)

        dataset = miserables(True)
        self.assertEqual(dataset.adjacency.shape, (77, 77))
        self.assertEqual(len(dataset.names), 77)

    def test_directed(self):
        adjacency = painters()
        self.assertEqual(adjacency.shape, (14, 14))

        adjacency = art_philo_science()
        self.assertEqual(adjacency.shape, (30, 30))

        dataset = painters(True)
        self.assertEqual(dataset.adjacency.shape, (14, 14))
        self.assertEqual(len(dataset.names), 14)

        dataset = art_philo_science(True)
        self.assertEqual(dataset.adjacency.shape, (30, 30))
        self.assertEqual(len(dataset.names), 30)

    def test_bipartite(self):
        dataset = star_wars(True)
        self.assertEqual(dataset.biadjacency.shape, (4, 3))
        self.assertEqual(len(dataset.names), 4)
        self.assertEqual(len(dataset.names_col), 3)

        dataset = movie_actor(True)
        self.assertEqual(dataset.biadjacency.shape, (15, 17))
        self.assertEqual(len(dataset.names), 15)
        self.assertEqual(len(dataset.names_col), 17)

        dataset = hourglass(True)
        self.assertEqual(dataset.biadjacency.shape, (2, 2))

        dataset = art_philo_science(True)
        self.assertEqual(dataset.biadjacency.shape, (30, 11))
        self.assertEqual(len(dataset.names), 30)
        self.assertEqual(len(dataset.names_col), 11)
