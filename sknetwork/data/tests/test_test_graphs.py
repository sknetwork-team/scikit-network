# -*- coding: utf-8 -*-
# tests for test_graphs.py
"""
@author: Quentin Lutz <qlutz@enst.fr>
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <tbonald@enst.fr>
"""

import unittest

from sknetwork.data.test_graphs import *


class TestTestGraphs(unittest.TestCase):

    def test_undirected(self):
        adjacency = test_graph()
        self.assertEqual(adjacency.shape, (10, 10))
        adjacency = test_disconnected_graph()
        self.assertEqual(adjacency.shape, (10, 10))

    def test_directed(self):
        adjacency = test_digraph()
        self.assertEqual(adjacency.shape, (10, 10))

    def test_bipartite(self):
        biadjacency = test_bigraph()
        self.assertEqual(biadjacency.shape, (4, 5))
        biadjacency = test_disconnected_bigraph()
        self.assertEqual(biadjacency.shape, (4, 5))
