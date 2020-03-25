#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.hierarchy import Ward, BiWard
from sknetwork.embedding import Spectral
from sknetwork.data import karate_club, painters, movie_actor


class TestWard(unittest.TestCase):

    def setUp(self):
        self.ward = Ward()
        self.biward = BiWard(cluster_col=True, cluster_both=True)

    def test_undirected(self):
        adjacency = karate_club()
        dendrogram = self.ward.fit_transform(adjacency)
        self.assertEqual(dendrogram.shape, (adjacency.shape[0] - 1, 4))

    def test_directed(self):
        adjacency = painters()
        dendrogram = self.ward.fit_transform(adjacency)
        self.assertEqual(dendrogram.shape, (adjacency.shape[0] - 1, 4))

    def test_bipartite(self):
        biadjacency = movie_actor()
        self.biward.fit(biadjacency)
        n1, n2 = biadjacency.shape
        self.assertEqual(self.biward.dendrogram_.shape, (n1 - 1, 4))
        self.assertEqual(self.biward.dendrogram_row_.shape, (n1 - 1, 4))
        self.assertEqual(self.biward.dendrogram_col_.shape, (n2 - 1, 4))
        self.assertEqual(self.biward.dendrogram_full_.shape, (n1 + n2 - 1, 4))

    def test_disconnected(self):
        adjacency = np.eye(10)
        dendrogram = self.ward.fit_transform(adjacency)
        self.assertEqual(dendrogram.shape, (9, 4))

    def test_options(self):
        adjacency = karate_club()
        ward = Ward(embedding_method=Spectral())
        dendrogram = ward.fit_transform(adjacency)
        self.assertEqual(dendrogram.shape, (adjacency.shape[0] - 1, 4))

    # noinspection PyTypeChecker
    def test_input(self):
        with self.assertRaises(TypeError):
            self.ward.fit(sparse.identity(1))
