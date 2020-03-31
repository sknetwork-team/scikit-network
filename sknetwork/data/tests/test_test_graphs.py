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
        graph = Simple()
        self.assertEqual(graph.adjacency.shape, (10, 10))
        graph = DisSimple()
        self.assertEqual(graph.adjacency.shape, (10, 10))

    def test_directed(self):
        graph = DiSimple()
        self.assertEqual(graph.adjacency.shape, (10, 10))

    def test_bipartite(self):
        graph = BiSimple()
        self.assertEqual(graph.biadjacency.shape, (4, 5))
        graph = BiDisSimple()
        self.assertEqual(graph.biadjacency.shape, (4, 5))
