#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for clustering API"""
import unittest

from sknetwork.clustering import *
from sknetwork.data import house, star_wars
from sknetwork.data.test_graphs import *
from sknetwork.embedding.svd import GSVD


class TestClusteringAPI(unittest.TestCase):

    def test_regular(self):

        clustering = [Louvain(return_aggregate=True), KMeans(embedding_method=GSVD(3), return_aggregate=True),
                      PropagationClustering(return_aggregate=True)]
        for clustering_algo in clustering:
            for adjacency in [test_graph(), test_digraph(), test_graph_disconnect()]:
                n = adjacency.shape[0]
                adjacency_bool = adjacency.copy()
                adjacency_bool.data = adjacency_bool.data.astype(bool)

                labels1 = clustering_algo.fit_transform(adjacency)
                labels2 = clustering_algo.fit_transform(adjacency_bool)
                self.assertEqual(labels1.shape, (n,))
                self.assertEqual(labels2.shape, (n,))

                n_labels = len(set(labels2))
                self.assertEqual(clustering_algo.adjacency_.shape, (n_labels, n_labels))

    def test_bipartite(self):
        biadjacency = test_bigraph()
        n_row, n_col = biadjacency.shape

        clustering = [BiLouvain(return_aggregate=True),
                      BiKMeans(embedding_method=GSVD(3), co_cluster=True, return_aggregate=True),
                      BiPropagationClustering(return_aggregate=True)]
        for clustering_algo in clustering:
            clustering_algo.fit_transform(biadjacency)
            labels_row = clustering_algo.labels_row_
            labels_col = clustering_algo.labels_col_
            self.assertEqual(labels_row.shape, (n_row,))
            self.assertEqual(labels_col.shape, (n_col,))

    def test_aggregate(self):
        adjacency = house()
        biadjacency = star_wars()

        louvain = Louvain(return_aggregate=True)
        louvain.fit(adjacency)
        self.assertTrue(np.issubdtype(louvain.adjacency_.dtype, np.float_))

        bilouvain = BiLouvain(return_aggregate=True)
        bilouvain.fit(biadjacency)
        self.assertTrue(np.issubdtype(bilouvain.biadjacency_.dtype, np.float_))
