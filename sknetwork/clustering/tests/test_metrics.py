# -*- coding: utf-8 -*-
# tests for metrics.py
""""tests for clustering metrics"""
import unittest

import numpy as np

from sknetwork.clustering import modularity, bimodularity, comodularity, normalized_std, Louvain
from sknetwork.data import star_wars, karate_club
from sknetwork.data.test_graphs import test_graph


class TestClusteringMetrics(unittest.TestCase):

    def setUp(self):
        """Basic graph for tests"""
        self.adjacency = test_graph()
        n = self.adjacency.shape[0]
        labels = np.zeros(n)
        labels[0] = 1
        self.labels = labels.astype(int)
        self.unique_cluster = np.zeros(n, dtype=int)

    def test_api(self):
        for metric in [modularity, comodularity]:
            _, fit, div = metric(self.adjacency, self.labels, return_all=True)
            mod = metric(self.adjacency, self.labels, return_all=False)
            self.assertAlmostEqual(fit - div, mod)
            self.assertAlmostEqual(metric(self.adjacency, self.unique_cluster), 0.)

            with self.assertRaises(ValueError):
                metric(self.adjacency, self.labels[:3])

    def test_modularity(self):
        adjacency = karate_club()
        labels = Louvain().fit_transform(adjacency)
        self.assertAlmostEqual(0.42, modularity(adjacency, labels), places=2)

    def test_bimodularity(self):
        biadjacency = star_wars()
        labels_row = np.array([0, 0, 1, 1])
        labels_col = np.array([0, 1, 0])
        self.assertEqual(bimodularity(biadjacency, labels_row, labels_col), 0.1875)

        with self.assertRaises(ValueError):
            bimodularity(biadjacency, labels_row[:2], labels_col)
        with self.assertRaises(ValueError):
            bimodularity(biadjacency, labels_row, labels_col[:2])

    def test_nsd(self):
        balanced = np.arange(5)
        self.assertEqual(1, normalized_std(balanced))
        unbalanced = np.zeros(5)
        unbalanced[0] = 1
        self.assertGreaterEqual(normalized_std(unbalanced), 0)

        with self.assertRaises(ValueError):
            normalized_std(np.zeros(5))
