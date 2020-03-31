# -*- coding: utf-8 -*-
# tests for toys.py
"""
@author: Quentin Lutz <qlutz@enst.fr>
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <tbonald@enst.fr>
"""

import unittest

from sknetwork.data.toys import House, BowTie, KarateClub, Miserables, Painters, StarWars, MovieActor


class TestToys(unittest.TestCase):

    def test_undirected(self):
        graph = House()
        self.assertEqual(graph.adjacency.shape, (5, 5))

        graph = BowTie()
        self.assertEqual(graph.adjacency.shape, (5, 5))

        graph = KarateClub()
        self.assertEqual(graph.adjacency.shape, (34, 34))
        self.assertEqual(len(graph.labels), 34)

        graph = Miserables()
        self.assertEqual(graph.adjacency.shape, (77, 77))
        self.assertEqual(len(graph.names), 77)

    def test_directed(self):
        graph = Painters()
        self.assertEqual(graph.adjacency.shape, (14, 14))
        self.assertEqual(len(graph.names), 14)

    def test_bipartite(self):
        graph = StarWars()
        self.assertEqual(graph.biadjacency.shape, (4, 3))
        self.assertEqual(len(graph.names), 4)
        self.assertEqual(len(graph.names_col), 3)

        graph = MovieActor()
        self.assertEqual(graph.biadjacency.shape, (15, 16))
        self.assertEqual(len(graph.names), 15)
        self.assertEqual(len(graph.names_col), 16)
