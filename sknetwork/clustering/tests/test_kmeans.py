#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest

from sknetwork.clustering import BiKMeans, KMeans
from sknetwork.data.basic import *
from sknetwork.data import KarateClub, MovieActor
from sknetwork.embedding import BiSpectral, Spectral


class TestKMeans(unittest.TestCase):

    def setUp(self):
        self.kmeans = KMeans(n_clusters=3)
        self.bikmeans = BiKMeans(n_clusters=3, cluster_both=True)

    def test_undirected(self):
        adjacency = Small().adjacency
        labels = self.kmeans.fit_transform(adjacency)
        self.assertEqual(len(set(labels)), 3)

    def test_directed(self):
        adjacency = DiSmall().adjacency
        labels = self.kmeans.fit_transform(adjacency)
        self.assertEqual(len(set(labels)), 3)

    def test_bipartite(self):
        biadjacency = BiSmall().biadjacency
        self.bikmeans.fit(biadjacency)
        labels = np.hstack((self.bikmeans.labels_row_, self.bikmeans.labels_col_))
        self.assertEqual(len(set(labels)), 3)

    def test_disconnected(self):
        adjacency = SmallDisconnected().adjacency
        self.kmeans.n_clusters = 5
        labels = self.kmeans.fit_transform(adjacency)
        self.assertTrue(len(set(labels)), 5)

    def test_options(self):
        adjacency = KarateClub().adjacency

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
        biadjacency = MovieActor().biadjacency

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

    def test_unknown_types(self):
        with self.assertRaises(TypeError):
            self.kmeans.fit(sparse.identity(10))

