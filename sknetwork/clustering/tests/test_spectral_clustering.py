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
        sc = SpectralClustering(embedding_dimension=3)
        sc.fit(self.undirected)
        self.assertEqual(sc.labels_.shape[0], self.undirected.shape[0])

    def test_bipartite(self):
        n_clusters = 2
        n1, n2 = self.bipartite.shape
        sc = SpectralClustering(embedding_dimension=3, n_clusters=n_clusters, co_clustering=True)
        sc.fit(self.bipartite)
        self.assertEqual(len(sc.labels_), n1)
        self.assertEqual(len(sc.col_labels_), n2)
