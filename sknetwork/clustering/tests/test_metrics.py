# -*- coding: utf-8 -*-
# tests for metrics.py
""""tests for clustering metrics"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.clustering import modularity, bimodularity, cocitation_modularity, nsd
from sknetwork.data import star_wars_villains


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
        self.star_wars_graph: sparse.csr_matrix = star_wars_villains()
        self.villain_labels = np.array([0, 0, 1, 1])
        self.movie_labels = np.array([0, 1, 0])
        self.assertEqual(bimodularity(self.star_wars_graph, self.villain_labels, self.movie_labels), 0.1875)

    def test_cocitation_modularity(self):
        self.assertAlmostEqual(cocitation_modularity(self.graph, self.labels), 0.0521, 3)
        self.assertAlmostEqual(cocitation_modularity(self.graph, self.unique_cluster), 0.)

    def test_nsd(self):
        balanced = np.arange(5)
        self.assertEqual(1, nsd(balanced))
        unbalanced = np.zeros(5)
        unbalanced[0] = 1
        self.assertGreaterEqual(nsd(unbalanced), 0)
