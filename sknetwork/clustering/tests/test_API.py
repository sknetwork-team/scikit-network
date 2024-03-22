#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for clustering API"""
import unittest

from sknetwork.clustering import *
from sknetwork.data.test_graphs import *


class TestClusteringAPI(unittest.TestCase):

    def setUp(self):
        self.algos = [Louvain(return_aggregate=True), Leiden(return_aggregate=True),
                      PropagationClustering(return_aggregate=True)]

    def test_regular(self):
        for algo in self.algos:
            for adjacency in [test_graph(), test_digraph(), test_disconnected_graph()]:
                n = adjacency.shape[0]
                labels = algo.fit_predict(adjacency)
                n_labels = len(set(labels))
                self.assertEqual(labels.shape, (n,))
                self.assertEqual(algo.aggregate_.shape, (n_labels, n_labels))
                adjacency_bool = adjacency.astype(bool)
                labels = algo.fit_predict(adjacency_bool)
                n_labels = len(set(labels))
                self.assertEqual(labels.shape, (n,))
                self.assertEqual(algo.aggregate_.shape, (n_labels, n_labels))
                membership = algo.fit_transform(adjacency_bool)
                self.assertEqual(membership.shape, (n, n_labels))

    def test_bipartite(self):
        biadjacency = test_bigraph()
        n_row, n_col = biadjacency.shape
        for algo in self.algos:
            algo.fit(biadjacency)
            self.assertEqual(algo.labels_row_.shape, (n_row,))
            self.assertEqual(algo.labels_col_.shape, (n_col,))
