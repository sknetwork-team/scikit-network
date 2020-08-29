# -*- coding: utf-8 -*-
# tests for toy_graphs.py
"""
@author: Quentin Lutz <qlutz@enst.fr>
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <tbonald@enst.fr>
"""
import unittest

from sknetwork.data.toy_graphs import *


class TestToys(unittest.TestCase):

    def test_undirected(self):
        adjacency = house()
        self.assertEqual(adjacency.shape, (5, 5))

        graph = house(metadata=True)
        self.assertEqual(graph.position.shape, (5, 2))

        adjacency = bow_tie()
        self.assertEqual(adjacency.shape, (5, 5))

        graph = bow_tie(metadata=True)
        self.assertEqual(graph.position.shape, (5, 2))

        graph = karate_club(True)
        self.assertEqual(graph.adjacency.shape, (34, 34))
        self.assertEqual(len(graph.labels), 34)

        graph = miserables(True)
        self.assertEqual(graph.adjacency.shape, (77, 77))
        self.assertEqual(len(graph.names), 77)

    def test_directed(self):
        adjacency = painters()
        self.assertEqual(adjacency.shape, (14, 14))

        graph = painters(True)
        self.assertEqual(graph.adjacency.shape, (14, 14))
        self.assertEqual(len(graph.names), 14)

    def test_bipartite(self):
        graph = star_wars(True)
        self.assertEqual(graph.biadjacency.shape, (4, 3))
        self.assertEqual(len(graph.names), 4)
        self.assertEqual(len(graph.names_col), 3)

        graph = movie_actor(True)
        self.assertEqual(graph.biadjacency.shape, (15, 16))
        self.assertEqual(len(graph.names), 15)
        self.assertEqual(len(graph.names_col), 16)

        graph = hourglass(True)
        self.assertEqual(graph.biadjacency.shape, (2, 2))
