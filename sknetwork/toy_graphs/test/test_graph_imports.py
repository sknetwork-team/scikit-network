# -*- coding: utf-8 -*-
# test for graph_data.py
#
# Copyright 2018 Scikit-network Developers.
# Copyright 2018 Quentin Lutz <qlutz@enst.fr>, Nathan de Lara <ndelara@enst.fr>
#
# This file is part of Scikit-network.

import unittest
from sknetwork.toy_graphs.graph_data import karate_club_graph, star_wars_villains_graph


class TestGraphImport(unittest.TestCase):

    def setUp(self):
        pass

    def test_available(self):
        graph = karate_club_graph()
        self.assertEqual(graph.shape[0], 34)

        graph = star_wars_villains_graph()
        self.assertEqual(graph.shape, (4, 3))
