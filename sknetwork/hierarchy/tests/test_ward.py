#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest

from scipy import sparse

from sknetwork.hierarchy import Ward, BiWard, straight_cut, balanced_cut
from sknetwork.embedding import Spectral
from sknetwork.data import karate_club, painters, movie_actor


class TestWard(unittest.TestCase):

    def setUp(self):
        self.ward = Ward()
        self.biward = BiWard(cluster_col=True, cluster_both=True)

        self.adjacency: sparse.csr_matrix = karate_club()
        self.adjacency_: sparse.csr_matrix = painters()
        self.biadjacency: sparse.csr_matrix = movie_actor()

    def test_undirected(self):
        dendrogram = self.ward.fit_transform(self.adjacency)
        self.assertEqual(dendrogram.shape, (self.adjacency.shape[0] - 1, 4))
        labels = straight_cut(dendrogram, sorted_clusters=True)
        self.assertEqual(len(set(labels)), 2)
        labels = balanced_cut(dendrogram, max_cluster_size=10)
        self.assertEqual(len(set(labels)), 6)

    def test_directed(self):
        dendrogram = self.ward.fit_transform(self.adjacency_)
        self.assertEqual(dendrogram.shape, (self.adjacency_.shape[0] - 1, 4))

    def test_bipartite(self):
        self.biward.fit(self.biadjacency)
        n1, n2 = self.biadjacency.shape
        self.assertEqual(self.biward.dendrogram_.shape, (n1 - 1, 4))
        self.assertEqual(self.biward.dendrogram_row_.shape, (n1 - 1, 4))
        self.assertEqual(self.biward.dendrogram_col_.shape, (n2 - 1, 4))
        self.assertEqual(self.biward.dendrogram_full_.shape, (n1 + n2 - 1, 4))

    def test_options(self):
        ward = Ward(embedding_method=Spectral())
        dendrogram = ward.fit_transform(self.adjacency)
        self.assertEqual(dendrogram.shape, (self.adjacency.shape[0] - 1, 4))

    # noinspection PyTypeChecker
    def test_unknown_types(self):
        with self.assertRaises(TypeError):
            self.ward.fit(sparse.identity(1))
