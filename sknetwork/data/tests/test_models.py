# -*- coding: utf-8 -*-
# tests for toys.py
"""authors: Quentin Lutz <qlutz@enst.fr>"""

import unittest

import numpy as np

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
        adj, g_t_f, g_t_s = block_model(2, shape=(2, 2), random_state=1)
        self.assertEqual((adj.indptr == [0, 0, 1]).all(), True)
        self.assertEqual((adj.indices == [0]).all(), True)
        self.assertEqual((adj.data == [True]).all(), True)
        self.assertEqual((g_t_f == [0, 1]).all(), True)
        self.assertEqual((g_t_s == [0, 1]).all(), True)

        adj, g_t_f, g_t_s = block_model(np.ones((2, 2), dtype=int), shape=(2, 2), random_state=1)
        self.assertEqual((adj.indptr == [0, 0, 1]).all(), True)
        self.assertEqual((adj.indices == [0]).all(), True)
        self.assertEqual((adj.data == [True]).all(), True)
        self.assertEqual((g_t_f == [0, 1]).all(), True)
        self.assertEqual((g_t_s == [0, 1]).all(), True)
