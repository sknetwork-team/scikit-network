#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest

from sknetwork.clustering import KMeans
from sknetwork.data.test_graphs import *
from sknetwork.embedding import GSVD, Spectral


class TestKMeans(unittest.TestCase):

    def test_undirected(self):
        n_clusters = 3
        algo = KMeans(n_clusters, GSVD(2))
        algo_options = KMeans(n_clusters, Spectral(3), co_cluster=True, sort_clusters=False)
        for adjacency in [test_graph(), test_graph_disconnect(), test_digraph()]:
            n = adjacency.shape[0]
            labels = algo.fit_transform(adjacency)
            self.assertEqual(len(set(labels)), n_clusters)
            self.assertEqual(algo.membership_.shape, (n, n_clusters))
            self.assertEqual(algo.aggregate_.shape, (n_clusters, n_clusters))
            labels = algo_options.fit_transform(adjacency)
            self.assertEqual(len(set(labels)), n_clusters)

    def test_bipartite(self):
        algo = KMeans(3, GSVD(2))
        algo_options = KMeans(4, Spectral(3), co_cluster=True, sort_clusters=False)
        for biadjacency in [test_bigraph(), test_bigraph_disconnect()]:
            n_row, n_col = biadjacency.shape
            algo.fit(biadjacency)
            self.assertEqual(len(algo.labels_), n_row)
            self.assertEqual(algo.membership_.shape, (n_row, 3))
            self.assertEqual(algo.membership_row_.shape, (n_row, 3))
            self.assertEqual(algo.membership_col_.shape, (n_col, 3))
            self.assertEqual(algo.aggregate_.shape, (3, 3))
            algo_options.fit(biadjacency)
            labels = np.hstack((algo_options.labels_row_, algo_options.labels_col_))
            self.assertEqual(len(set(labels)), 4)
            self.assertEqual(algo_options.membership_.shape, (n_row, 4))
            self.assertEqual(algo_options.membership_row_.shape, (n_row, 4))
            self.assertEqual(algo_options.membership_col_.shape, (n_col, 4))
            self.assertEqual(algo_options.aggregate_.shape, (4, 4))

