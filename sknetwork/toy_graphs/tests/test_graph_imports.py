# -*- coding: utf-8 -*-
# tests for graph_data.py
# authors: Quentin Lutz <qlutz@enst.fr>, Nathan de Lara <ndelara@enst.fr>

import unittest
from sknetwork.toy_graphs.graph_data import karate_club, star_wars_villains, \
    movie_actor, painters


class TestGraphImport(unittest.TestCase):

    def setUp(self):
        pass

    def test_available(self):
        adjacency = karate_club()
        self.assertEqual(adjacency.shape[0], 34)

        biadjacency = star_wars_villains()
        self.assertEqual(biadjacency.shape, (4, 3))

        biadjacency = movie_actor()
        self.assertEqual(biadjacency.shape, (15, 16))

        adjacency = painters()
        self.assertEqual(adjacency.shape[0], 14)
