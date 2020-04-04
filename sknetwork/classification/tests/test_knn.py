#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for KNN classification"""

import unittest

from sknetwork.classification import KNN, BiKNN
from sknetwork.data.test_graphs import *
from sknetwork.embedding import GSVD


# noinspection DuplicatedCode
class TestKNN(unittest.TestCase):

    def test_undirected(self):
        adjacency = test_graph()
        seeds_array = -np.ones(adjacency.shape[0])
        seeds_array[:2] = np.arange(2)
        seeds_dict = {0: 0, 1: 1}

        knn = KNN(embedding_method=GSVD(3), n_neighbors=2)
        labels_array = knn.fit_transform(adjacency, seeds_array)
        labels_dict = knn.fit_transform(adjacency, seeds_dict)
        self.assertTrue(np.allclose(labels_array, labels_dict))
        self.assertEqual(labels_array.shape[0], adjacency.shape[0])

    def test_bipartite(self):
        biadjacency = test_bigraph()
        n_row, n_col = biadjacency.shape
        seeds_row = {0: 0, 1: 1, 2: 0}
        seeds_col = {0: 0, 1: 1}

        biknn = BiKNN(embedding_method=GSVD(3), n_neighbors=2)
        labels = biknn.fit_transform(biadjacency, seeds_row)
        self.assertEqual(len(labels), n_row)
        self.assertEqual(len(biknn.labels_row_), n_row)
        self.assertEqual(len(biknn.labels_col_), n_col)
        labels = biknn.fit_transform(biadjacency, seeds_row, seeds_col)
        self.assertEqual(len(labels), n_row)
        self.assertEqual(len(biknn.labels_row_), n_row)
        self.assertEqual(len(biknn.labels_col_), n_col)
