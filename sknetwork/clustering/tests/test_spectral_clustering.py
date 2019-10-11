#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest

from scipy import sparse

from sknetwork.clustering import SpectralClustering
from sknetwork.embedding import SVD
from sknetwork.toy_graphs import karate_club, painters, movie_actor
from sknetwork.utils import KMeans


class TestSpectralClustering(unittest.TestCase):

    def setUp(self):
        self.undirected: sparse.csr_matrix = karate_club()
        self.directed: sparse.csr_matrix = painters()
        self.bipartite: sparse.csr_matrix = movie_actor()

    def test_default_options(self):
        sc = SpectralClustering(embedding_dimension=3, embedding_algo='spectral')
        sc.fit(self.undirected)
        self.assertEqual(sc.labels_.shape[0], self.undirected.shape[0])

        sc = SpectralClustering(embedding_dimension=3, embedding_algo='svd')
        sc.fit(self.directed)
        self.assertEqual(sc.labels_.shape[0], self.directed.shape[0])

    def test_custom_options(self):
        n_clusters = 2
        sc = SpectralClustering(clustering_algo=KMeans(n_clusters=n_clusters),
                                embedding_algo=SVD(embedding_dimension=4))
        sc.fit(self.bipartite)
        self.assertEqual(len(set(sc.labels_)), n_clusters)
