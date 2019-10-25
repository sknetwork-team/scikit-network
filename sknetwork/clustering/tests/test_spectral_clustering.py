#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest

from scipy import sparse

from sknetwork.clustering import BiSpectralClustering, SpectralClustering
from sknetwork.toy_graphs import karate_club, painters, movie_actor


class TestSpectralClustering(unittest.TestCase):

    def setUp(self):
        self.undirected: sparse.csr_matrix = karate_club()
        self.directed: sparse.csr_matrix = painters()
        self.bipartite: sparse.csr_matrix = movie_actor()

    def test_default_options(self):
        sc = SpectralClustering(embedding_dimension=3)
        sc.fit(self.undirected)
        self.assertEqual(sc.labels_.shape[0], self.undirected.shape[0])

    def test_biclustering(self):
        bsc = BiSpectralClustering(embedding_dimension=3, joint_clustering=False)
        bsc.fit(self.directed)
        self.assertEqual(bsc.labels_.shape[0], self.directed.shape[0])
        self.assertTrue(bsc.col_labels_ is None)

    def test_joint_clustering(self):
        bsc = BiSpectralClustering(embedding_dimension=3)
        bsc.fit(self.bipartite)
        self.assertEqual(bsc.labels_.shape[0], self.bipartite.shape[0])
        self.assertEqual(bsc.col_labels_.shape[0], self.bipartite.shape[1])
