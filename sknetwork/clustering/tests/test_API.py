#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for clustering API"""

import unittest

from sknetwork.clustering import *
from sknetwork.data.test_graphs import *
from sknetwork.embedding.svd import GSVD


class TestClusteringAPI(unittest.TestCase):

    def test_undirected(self):
        adjacency = test_graph()
        n = adjacency.shape[0]

        clustering = [Louvain(), KMeans(embedding_method=GSVD(3))]
        for clustering_algo in clustering:
            labels = clustering_algo.fit_transform(adjacency)
            self.assertEqual(len(labels), n)

    def test_bipartite(self):
        biadjacency = test_bigraph()
        n_row, n_col = biadjacency.shape

        clustering = [BiLouvain(), BiKMeans(embedding_method=GSVD(3), co_cluster=True)]
        for clustering_algo in clustering:
            clustering_algo.fit_transform(biadjacency)
            self.assertEqual(len(clustering_algo.labels_), n_row)
            self.assertEqual(len(clustering_algo.labels_col_), n_col)
