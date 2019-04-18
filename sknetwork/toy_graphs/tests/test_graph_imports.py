# -*- coding: utf-8 -*-
# tests for graph_data.py
# authors: Quentin Lutz <qlutz@enst.fr>, Nathan de Lara <ndelara@enst.fr>

import unittest
from sknetwork.toy_graphs.graph_data import karate_club_graph, star_wars_villains_graph, movie_actor_graph


class TestGraphImport(unittest.TestCase):

    def setUp(self):
        pass

    def test_available(self):
        graph = karate_club_graph()
        self.assertEqual(graph.shape[0], 34)

        graph = star_wars_villains_graph()
        self.assertEqual(graph.shape, (4, 3))

        graph = movie_actor_graph()
        self.assertEqual(graph.shape, (15, 16))
