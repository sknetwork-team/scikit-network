# -*- coding: utf-8 -*-
# test for graph_data.py
#
# Copyright 2018 Scikit-network Developers.
# Copyright 2018 Quentin Lutz <qlutz@enst.fr>
#
# This file is part of Scikit-network.

import unittest
from sknetwork.toy_graphs.graph_data import GraphConstants


class TestGraphImport(unittest.TestCase):

    def setUp(self):
        pass

    def test_available(self):
        graph = GraphConstants.karate_club_graph()
        self.assertEqual(graph.shape[0], 34)
