# -*- coding: utf-8 -*-
# tests for metrics.py
""""tests for clustering metrics"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.clustering import modularity, bimodularity, comodularity, normalized_std
from sknetwork.data import star_wars


# noinspection PyMissingOrEmptyDocstring
class TestClusteringMetrics(unittest.TestCase):

    def setUp(self):
        self.graph = sparse.csr_matrix(np.array([[0, 1, 1, 1],
                                                 [1, 0, 0, 0],
                                                 [1, 0, 0, 1],
                                                 [1, 0, 1, 0]]))
        self.labels = np.array([0, 1, 0, 0])
        self.unique_cluster = np.zeros(4, dtype=int)

    def test_modularity(self):
        self.assertAlmostEqual(modularity(self.graph, self.labels), -0.0312, 3)
        self.assertAlmostEqual(modularity(self.graph, self.unique_cluster), 0.)

    def test_bimodularity(self):
        biadjacency = star_wars()
        labels_row = np.array([0, 0, 1, 1])
        labels_col = np.array([0, 1, 0])
        self.assertEqual(bimodularity(biadjacency, labels_row, labels_col), 0.1875)

    def test_cocitation_modularity(self):
        self.assertAlmostEqual(comodularity(self.graph, self.labels), 0.0521, 3)
        self.assertAlmostEqual(comodularity(self.graph, self.unique_cluster), 0.)

    def test_nsd(self):
        balanced = np.arange(5)
        self.assertEqual(1, normalized_std(balanced))
        unbalanced = np.zeros(5)
        unbalanced[0] = 1
        self.assertGreaterEqual(normalized_std(unbalanced), 0)
