#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for KNN classification"""

import unittest

from sknetwork.classification import KNN
from sknetwork.data.test_graphs import *


# noinspection DuplicatedCode
class TestKNN(unittest.TestCase):

    def test_undirected(self):
        adjacency = test_graph()
        adj_array_seeds = -np.ones(adjacency.shape[0])
        adj_array_seeds[:2] = np.arange(2)
        adj_dict_seeds = {0: 0, 1: 1}

        knn = KNN()
        labels1 = knn.fit_transform(adjacency, adj_array_seeds)
        labels2 = knn.fit_transform(adjacency, adj_dict_seeds)
        self.assertTrue(np.allclose(labels1, labels2))
        self.assertEqual(labels2.shape[0], adjacency.shape[0])
