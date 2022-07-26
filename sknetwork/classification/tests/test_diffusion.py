#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for DiffusionClassifier"""

import unittest

from sknetwork.classification import DiffusionClassifier
from sknetwork.data.test_graphs import *


class TestDiffusionClassifier(unittest.TestCase):

    def test_graph(self):
        adjacency = test_graph()
        seeds = {0: 0, 1: 1}
        algo = DiffusionClassifier()
        algo.fit(adjacency, seeds=seeds)
        self.assertTrue(len(algo.labels_) == adjacency.shape[0])
        adjacency = test_digraph()
        algo = DiffusionClassifier(centering=False)
        algo.fit(adjacency, seeds=seeds)
        self.assertTrue(len(algo.labels_) == adjacency.shape[0])
        with self.assertRaises(ValueError):
            DiffusionClassifier(n_iter=0)
        algo = DiffusionClassifier(centering=False, threshold=1)
        algo.fit(adjacency, seeds=seeds)
        self.assertTrue(max(algo.labels_) == -1)

    def test_bipartite(self):
        biadjacency = test_bigraph()
        n_row, n_col = biadjacency.shape
        seeds_row = {0: 0, 1: 1}
        seeds_col = {5: 1}
        algo = DiffusionClassifier()
        algo.fit(biadjacency, seeds_row=seeds_row, seeds_col=seeds_col)
        self.assertTrue(len(algo.labels_row_) == n_row)
        self.assertTrue(len(algo.labels_col_) == n_col)
