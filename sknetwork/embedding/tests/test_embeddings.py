#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for embeddings"""

import unittest

import numpy as np

from sknetwork.embedding import Spectral, SVD
from sknetwork.linalg import SparseLR
from sknetwork.toy_graphs import random_graph, random_bipartite_graph, house


def barycenter(adjacency, embedding):
    weights = adjacency.dot(np.ones(adjacency.shape[1]))
    return embedding.T.dot(weights)


class TestEmbeddings(unittest.TestCase):

    def setUp(self):
        self.adjacency = random_graph()
        self.biadjacency = random_bipartite_graph()
        self.house = house()

    def test_spectral(self):
        spectral = Spectral(2)
        spectral.fit(self.adjacency)
        self.assertTrue(spectral.embedding_.shape == (self.adjacency.shape[0], 2))
        self.assertTrue(type(spectral.eigenvalues_) == np.ndarray and len(spectral.eigenvalues_) == 2)
        self.assertTrue(min(spectral.eigenvalues_ >= -1e-6))
        self.assertTrue(max(spectral.eigenvalues_ <= 2))

        spectral.fit(self.biadjacency)
        self.assertEqual(spectral.embedding_.shape, (self.biadjacency.shape[0], 2))
        self.assertEqual(spectral.coembedding_.shape, (self.biadjacency.shape[1], 2))
        self.assertTrue(type(spectral.eigenvalues_) == np.ndarray and len(spectral.eigenvalues_) == 2)

        # test if the embedding is centered
        # without regularization
        spectral.regularization = None
        spectral.fit(self.house)
        self.assertAlmostEqual(np.linalg.norm(barycenter(self.house, spectral.embedding_)), 0)

        # with regularization
        spectral.regularization = 1.
        n, m = self.house.shape
        slr = SparseLR(self.house, [(spectral.regularization * np.ones(n), np.ones(m))])
        spectral.fit(self.house)
        # self.assertAlmostEqual(np.linalg.norm(barycenter(slr, spectral.embedding_)), 0)

    def test_svd(self):
        svd = SVD(2)
        svd.fit(self.adjacency)
        self.assertTrue(svd.embedding_.shape == (self.adjacency.shape[0], 2))
        self.assertTrue(svd.coembedding_.shape == (self.adjacency.shape[0], 2))
        self.assertTrue(type(svd.singular_values_) == np.ndarray and len(svd.singular_values_) == 2)

        svd.fit(self.biadjacency)
        self.assertTrue(svd.embedding_.shape == (self.biadjacency.shape[0], 2))
        self.assertTrue(svd.coembedding_.shape == (self.biadjacency.shape[1], 2))
        self.assertTrue(type(svd.singular_values_) == np.ndarray and len(svd.singular_values_) == 2)
