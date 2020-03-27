#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.clustering import BiKMeans, KMeans
from sknetwork.data import karate_club, painters, movie_actor
from sknetwork.embedding import BiSpectral, Spectral


class TestKMeans(unittest.TestCase):

    def setUp(self):
        self.kmeans = KMeans()
        self.bikmeans = BiKMeans(cluster_both=True)

    def test_undirected(self):
        adjacency = karate_club()
        labels = self.kmeans.fit_transform(adjacency)
        self.assertEqual(len(set(labels)), 8)

    def test_directed(self):
        adjacency = painters()
        labels = self.kmeans.fit_transform(adjacency)
        self.assertEqual(len(set(labels)), 8)

    def test_bipartite(self):
        biadjacency = movie_actor()
        self.bikmeans.fit(biadjacency)
        labels = np.hstack((self.bikmeans.labels_row_, self.bikmeans.labels_col_))
        self.assertEqual(len(set(labels)), 8)

    def test_disconnected(self):
        adjacency = sparse.identity(2, format='csr')
        self.kmeans.n_clusters = 2
        labels = self.kmeans.fit_transform(adjacency)
        self.assertEqual(len(labels), 2)
        adjacency = sparse.identity(10, format='csr')
        self.kmeans.n_clusters = 5
        labels = self.kmeans.fit_transform(adjacency)
        self.assertTrue(len(set(labels)), 5)

    def test_options(self):
        adjacency = karate_club()
        kmeans = KMeans(embedding_method=Spectral())
        labels = kmeans.fit_transform(adjacency)
        self.assertTrue(len(set(labels)), 8)
        adjacency = movie_actor()
        kmeans = KMeans(embedding_method=BiSpectral())
        labels = kmeans.fit_transform(adjacency)
        self.assertTrue(len(set(labels)), 8)

    def test_unknown_types(self):
        with self.assertRaises(TypeError):
            self.kmeans.fit(sparse.identity(10))

