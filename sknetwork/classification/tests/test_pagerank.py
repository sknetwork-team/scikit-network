#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for PageRankClassifier"""

import unittest

from sknetwork.classification import PageRankClassifier, BiPageRankClassifier
from sknetwork.data.test_graphs import *


# noinspection DuplicatedCode
class TestPageRankClassifier(unittest.TestCase):

    def test_undirected(self):
        adjacency = test_graph()
        adj_array_seeds = -np.ones(adjacency.shape[0])
        adj_array_seeds[:2] = np.arange(2)
        adj_dict_seeds = {0: 0, 1: 1}

        mr = PageRankClassifier(solver='lsqr')
        labels1 = mr.fit_transform(adjacency, adj_array_seeds)
        labels2 = mr.fit_transform(adjacency, adj_dict_seeds)
        self.assertTrue(np.allclose(labels1, labels2))
        self.assertEqual(labels2.shape[0], adjacency.shape[0])

    def test_bipartite(self):
        biadjacency = test_bigraph()
        seeds_row_array = -np.ones(biadjacency.shape[0])
        seeds_row_array[:2] = np.arange(2)
        seeds_row_dict = {0: 0, 1: 1}
        seeds_col_dict = {0: 0}

        bmr = BiPageRankClassifier()
        labels1 = bmr.fit_transform(biadjacency, seeds_row_array)
        labels2 = bmr.fit_transform(biadjacency, seeds_row_dict)
        self.assertTrue(np.allclose(labels1, labels2))
        self.assertEqual(labels2.shape[0], biadjacency.shape[0])

        bmr.fit(biadjacency, seeds_row_dict, seeds_col_dict)
