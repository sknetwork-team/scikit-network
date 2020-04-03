# -*- coding: utf-8 -*-
# tests for toy_graphs.py
"""authors:
Thomas Bonald <bonald@enst.fr>
Quentin Lutz <qlutz@enst.fr>"""

import unittest

from sknetwork.data.models import *


class TestModels(unittest.TestCase):

    def test_undirected(self):
        adjacency = linear_graph(5)
        self.assertEqual(adjacency.shape, (5, 5))
        adjacency = cyclic_graph(5)
        self.assertEqual(adjacency.shape, (5, 5))
        adjacency = erdos_renyie(5)
        self.assertEqual(adjacency.shape, (5, 5))
        graph = block_model(np.array([4, 5, 6]), np.array([0.5, 0.3, 0.2]), 0.1, metadata=True)
        adjacency = graph.adjacency
        labels = graph.labels
        self.assertEqual(adjacency.shape, (15, 15))
        self.assertEqual(len(labels), 15)

    def test_directed(self):
        adjacency = linear_digraph(5)
        self.assertEqual(adjacency.shape, (5, 5))
        adjacency = cyclic_digraph(5)
        self.assertEqual(adjacency.shape, (5, 5))
