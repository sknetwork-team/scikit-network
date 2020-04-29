#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for clustering API"""

import unittest

from sknetwork.clustering import *
from sknetwork.data.test_graphs import *
from sknetwork.embedding.svd import GSVD


class TestClusteringAPI(unittest.TestCase):

    def test_regular(self):

        clustering = [Louvain(), KMeans(embedding_method=GSVD(3))]
        for clustering_algo in clustering:
            for adjacency in [test_graph(), test_digraph()]:
                n = adjacency.shape[0]
                labels = clustering_algo.fit_transform(adjacency)
                self.assertEqual(labels.shape, (n,))

    def test_bipartite(self):
        biadjacency = test_bigraph()
        n_row, n_col = biadjacency.shape

        clustering = [BiLouvain(), BiKMeans(embedding_method=GSVD(3), co_cluster=True)]
        for clustering_algo in clustering:
            clustering_algo.fit_transform(biadjacency)
            self.assertEqual(clustering_algo.labels_row_.shape, (n_row,))
            self.assertEqual(clustering_algo.labels_col_.shape, (n_col,))
