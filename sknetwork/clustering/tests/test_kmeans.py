#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest

from sknetwork.clustering import BiKMeans, KMeans
from sknetwork.data.test_graphs import *
from sknetwork.embedding import GSVD, SVD, BiSpectral


class TestKMeans(unittest.TestCase):

    def setUp(self):
        self.kmeans = KMeans(3, GSVD(2))
        self.bikmeans = BiKMeans(3, GSVD(2))
        self.kmeans_options = KMeans(4, SVD(3), sort_clusters=False)
        self.bikmeans_options = BiKMeans(4, BiSpectral(3), co_cluster=True, sort_clusters=False)

    def test_undirected(self):
        for adjacency in [test_graph(), test_graph_disconnect()]:
            n = adjacency.shape[0]
            labels = self.kmeans.fit_transform(adjacency)
            self.assertEqual(len(set(labels)), 3)
            self.assertEqual(self.kmeans.membership_.shape, (n, 3))
            self.assertEqual(self.kmeans.adjacency_.shape, (3, 3))
            labels = self.kmeans_options.fit_transform(adjacency)
            self.assertEqual(len(set(labels)), 4)

    def test_directed(self):
        adjacency = test_digraph()
        n = adjacency.shape[0]
        labels = self.kmeans.fit_transform(adjacency)
        self.assertEqual(len(set(labels)), 3)
        self.assertEqual(self.kmeans.membership_.shape, (n, 3))
        self.assertEqual(self.kmeans.adjacency_.shape, (3, 3))
        labels = self.kmeans_options.fit_transform(adjacency)
        self.assertEqual(len(set(labels)), 4)

    def test_bipartite(self):
        for biadjacency in [test_bigraph(), test_bigraph_disconnect()]:
            n_row, n_col = biadjacency.shape
            self.bikmeans.fit(biadjacency)
            self.assertEqual(len(set(self.bikmeans.labels_row_)), 3)
            self.assertEqual(self.bikmeans.membership_.shape, (n_row, 3))
            self.assertEqual(self.bikmeans.membership_row_.shape, (n_row, 3))
            self.assertEqual(self.bikmeans.biadjacency_.shape, (3, n_col))
            self.bikmeans_options.fit(biadjacency)
            labels = np.hstack((self.bikmeans_options.labels_row_, self.bikmeans_options.labels_col_))
            self.assertEqual(len(set(labels)), 4)
            self.assertEqual(self.bikmeans_options.membership_.shape, (n_row, 4))
            self.assertEqual(self.bikmeans_options.membership_row_.shape, (n_row, 4))
            self.assertEqual(self.bikmeans_options.membership_col_.shape, (n_col, 4))
            self.assertEqual(self.bikmeans_options.biadjacency_.shape, (4, 4))

