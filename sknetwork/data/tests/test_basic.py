# -*- coding: utf-8 -*-
# tests for toys.py
"""
@author: Quentin Lutz <qlutz@enst.fr>
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <tbonald@enst.fr>
"""

import unittest

from sknetwork.data.basic import *


class TestToys(unittest.TestCase):

    def test_undirected(self):
        graph = Small()
        self.assertEqual(graph.adjacency.shape, (10, 10))
        graph = SmallDisconnected()
        self.assertEqual(graph.adjacency.shape, (10, 10))

    def test_directed(self):
        graph = DiSmall()
        self.assertEqual(graph.adjacency.shape, (10, 10))

    def test_bipartite(self):
        graph = BiSmall()
        self.assertEqual(graph.biadjacency.shape, (4, 5))
        graph = BiSmallDisconnected()
        self.assertEqual(graph.biadjacency.shape, (4, 5))
