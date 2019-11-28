#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest

from scipy import sparse

from sknetwork.clustering import BiSpectralClustering, SpectralClustering
from sknetwork.data import karate_club, painters, movie_actor


class TestSpectralClustering(unittest.TestCase):

    def test_default_options(self):
        self.undirected: sparse.csr_matrix = karate_club()
        spectral_clustering = SpectralClustering(embedding_dimension=3)
        spectral_clustering.fit(self.undirected)
        self.assertEqual(spectral_clustering.labels_.shape[0], self.undirected.shape[0])

    def test_biclustering(self):
        self.directed: sparse.csr_matrix = painters()
        bispectral_clustering = BiSpectralClustering(embedding_dimension=3, co_clustering=False)
        bispectral_clustering.fit(self.directed)
        self.assertEqual(bispectral_clustering.labels_.shape[0], self.directed.shape[0])
        self.assertTrue(bispectral_clustering.col_labels_ is None)

    def test_co_clustering(self):
        self.bipartite: sparse.csr_matrix = movie_actor()
        n1, n2 = self.bipartite.shape
        bispectral_clustering = BiSpectralClustering(embedding_dimension=3)
        bispectral_clustering.fit(self.bipartite)
        self.assertEqual(bispectral_clustering.row_labels_.shape[0], n1)
        self.assertEqual(bispectral_clustering.col_labels_.shape[0], n2)
        self.assertEqual(bispectral_clustering.labels_.shape[0], n1 + n2)
