# -*- coding: utf-8 -*-
# tests for toy_graphs.py
"""authors: Quentin Lutz <qlutz@enst.fr>"""

import unittest

from sknetwork.data.models import *


class TestModels(unittest.TestCase):

    def test_undirected(self):
        graph = Line(5)
        self.assertEqual(graph.adjacency.shape, (5, 5))
        graph = Cycle(5)
        self.assertEqual(graph.adjacency.shape, (5, 5))

    def test_directed(self):
        graph = LineDirected(5)
        self.assertEqual(graph.adjacency.shape, (5, 5))
        graph = CycleDirected(5)
        self.assertEqual(graph.adjacency.shape, (5, 5))

    # noinspection DuplicatedCode
    def test_generation(self):
        graph = SBM(2, shape=(2, 2), random_state=1)
        biadjacency = graph.biadjacency
        labels = graph.labels
        labels_col = graph.labels_col
        self.assertEqual((biadjacency.indptr == [0, 0, 1]).all(), True)
        self.assertEqual((biadjacency.indices == [0]).all(), True)
        self.assertEqual((biadjacency.data == [True]).all(), True)
        self.assertEqual((labels_col == [0, 1]).all(), True)
        self.assertEqual((labels == [0, 1]).all(), True)

        graph = SBM(np.ones((2, 2), dtype=int), shape=(2, 2), random_state=1)
        biadjacency = graph.biadjacency
        labels = graph.labels
        labels_col = graph.labels_col
        self.assertEqual((biadjacency.indptr == [0, 0, 1]).all(), True)
        self.assertEqual((biadjacency.indices == [0]).all(), True)
        self.assertEqual((biadjacency.data == [True]).all(), True)
        self.assertEqual((labels_col == [0, 1]).all(), True)
        self.assertEqual((labels == [0, 1]).all(), True)
