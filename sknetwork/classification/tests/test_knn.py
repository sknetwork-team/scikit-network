#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for KNN classification"""

import unittest

from sknetwork.classification import KNN
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
        labels1 = knn.fit_transform(adjacency, seeds_array)
        labels2 = knn.fit_transform(adjacency, seeds_dict)
        self.assertTrue(np.allclose(labels1, labels2))
        self.assertEqual(labels2.shape[0], adjacency.shape[0])
