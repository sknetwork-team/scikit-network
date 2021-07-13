#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for clustering API"""
import unittest

from sknetwork.clustering import *
from sknetwork.data import house
from sknetwork.data.test_graphs import *
from sknetwork.embedding.svd import GSVD


class TestClusteringAPI(unittest.TestCase):

    def test_regular(self):
        for algo in [Louvain(return_aggregate=True), KMeans(embedding_method=GSVD(3), return_aggregate=True),
                     PropagationClustering(return_aggregate=True)]:
            for adjacency in [test_graph(), test_digraph(), test_graph_disconnect()]:
                n = adjacency.shape[0]
                labels = algo.fit_transform(adjacency)
                n_labels = len(set(labels))
                self.assertEqual(labels.shape, (n,))
                self.assertEqual(algo.aggregate_.shape, (n_labels, n_labels))
                adjacency_bool = adjacency.astype(bool)
                labels = algo.fit_transform(adjacency_bool)
                n_labels = len(set(labels))
                self.assertEqual(labels.shape, (n,))
                self.assertEqual(algo.aggregate_.shape, (n_labels, n_labels))

    def test_bipartite(self):
        biadjacency = test_bigraph()
        n_row, n_col = biadjacency.shape
        for algo in [Louvain(return_aggregate=True),
                     KMeans(embedding_method=GSVD(3), co_cluster=True, return_aggregate=True),
                     PropagationClustering(return_aggregate=True)]:
            algo.fit_transform(biadjacency)
            self.assertEqual(algo.labels_row_.shape, (n_row,))
            self.assertEqual(algo.labels_col_.shape, (n_col,))
