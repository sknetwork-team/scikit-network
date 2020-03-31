#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest

from sknetwork.clustering import BiKMeans, KMeans
from sknetwork.data.test_graphs import *
from sknetwork.data import karate_club, movie_actor
from sknetwork.embedding import BiSpectral, Spectral


class TestKMeans(unittest.TestCase):

    def setUp(self):
        self.kmeans = KMeans(n_clusters=3)
        self.bikmeans = BiKMeans(n_clusters=3, cluster_both=True)

    def test_undirected(self):
        adjacency = test_graph()
        labels = self.kmeans.fit_transform(adjacency)
        self.assertEqual(len(set(labels)), 3)

    def test_directed(self):
        adjacency = test_digraph()
        labels = self.kmeans.fit_transform(adjacency)
        self.assertEqual(len(set(labels)), 3)

    def test_bipartite(self):
        biadjacency = test_bigraph()
        self.bikmeans.fit(biadjacency)
        labels = np.hstack((self.bikmeans.labels_row_, self.bikmeans.labels_col_))
        self.assertEqual(len(set(labels)), 3)

    def test_disconnected(self):
        adjacency = test_disconnected_graph()
        self.kmeans.n_clusters = 5
        labels = self.kmeans.fit_transform(adjacency)
        self.assertTrue(len(set(labels)), 5)

    def test_options(self):
        adjacency = test_graph()

        # embedding
        kmeans = KMeans(embedding_method=Spectral())
        labels = kmeans.fit_transform(adjacency)
        self.assertTrue(len(set(labels)), 8)

        # aggregate graph
        kmeans = KMeans(return_graph=True)
        labels = kmeans.fit_transform(adjacency)
        n_labels = len(set(labels))
        self.assertEqual(kmeans.adjacency_.shape, (n_labels, n_labels))

    def test_options_bipartite(self):
        biadjacency = test_bigraph()

        # embedding
        bikmeans = BiKMeans(embedding_method=BiSpectral())
        labels = bikmeans.fit_transform(biadjacency)
        self.assertTrue(len(set(labels)), 8)

        # aggregate graph
        bikmeans = BiKMeans(cluster_both=True, return_graph=True)
        bikmeans.fit(biadjacency)
        n_labels_row = len(set(bikmeans.labels_row_))
        n_labels_col = len(set(bikmeans.labels_col_))
        self.assertEqual(bikmeans.biadjacency_.shape, (n_labels_row, n_labels_col))


