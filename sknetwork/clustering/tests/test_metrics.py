# -*- coding: utf-8 -*-
# tests for metrics.py
""""tests for clustering metrics"""
import unittest

import numpy as np

from sknetwork.clustering import get_modularity, Louvain
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
        for metric in [get_modularity]:
            _, fit, div = metric(self.adjacency, self.labels, return_all=True)
            mod = metric(self.adjacency, self.labels, return_all=False)
            self.assertAlmostEqual(fit - div, mod)
            self.assertAlmostEqual(metric(self.adjacency, self.unique_cluster), 0.)

            with self.assertRaises(ValueError):
                metric(self.adjacency, self.labels[:3])

    def test_modularity(self):
        adjacency = karate_club()
        labels = Louvain().fit_predict(adjacency)
        self.assertAlmostEqual(get_modularity(adjacency, labels), 0.42, 2)

    def test_bimodularity(self):
        biadjacency = star_wars()
        labels_row = np.array([0, 0, 1, 1])
        labels_col = np.array([0, 1, 0])
        self.assertAlmostEqual(get_modularity(biadjacency, labels_row, labels_col), 0.12, 2)

        with self.assertRaises(ValueError):
            get_modularity(biadjacency, labels_row)
        with self.assertRaises(ValueError):
            get_modularity(biadjacency, labels_row[:2], labels_col)
        with self.assertRaises(ValueError):
            get_modularity(biadjacency, labels_row, labels_col[:2])
