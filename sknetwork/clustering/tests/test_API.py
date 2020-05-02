#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for clustering API"""

import unittest

from sknetwork.clustering import *
from sknetwork.data.test_graphs import *
from sknetwork.embedding.svd import GSVD


class TestClusteringAPI(unittest.TestCase):

    def test_regular(self):

        clustering = [Louvain(), KMeans(embedding_method=GSVD(3)), PropagationClustering()]
        for clustering_algo in clustering:
            for adjacency in [test_graph(), test_digraph()]:
                n = adjacency.shape[0]
                adjacency_bool = adjacency.copy()
                adjacency_bool.data = adjacency_bool.data.astype(bool)

                labels1 = clustering_algo.fit_transform(adjacency)
                labels2 = clustering_algo.fit_transform(adjacency_bool)
                self.assertEqual(labels1.shape, (n,))
                self.assertEqual(labels2.shape, (n,))

    def test_bipartite(self):
        biadjacency = test_bigraph()
        n_row, n_col = biadjacency.shape

        clustering = [BiLouvain(), BiKMeans(embedding_method=GSVD(3), co_cluster=True),
                      BiPropagationClustering()]
        for clustering_algo in clustering:
            clustering_algo.fit_transform(biadjacency)
            self.assertEqual(clustering_algo.labels_row_.shape, (n_row,))
            self.assertEqual(clustering_algo.labels_col_.shape, (n_col,))
