#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for spectral embedding."""

import unittest

import numpy as np

from sknetwork.data.test_graphs import test_graph, test_bigraph, test_digraph, test_graph_disconnect
from sknetwork.embedding import Spectral, LaplacianEmbedding
from sknetwork.embedding.spectral import RegularizedLaplacian


class TestEmbeddings(unittest.TestCase):

    def setUp(self) -> None:
        """Simple case for testing"""
        self.adjacency = test_graph()
        self.n = self.adjacency.shape[0]
        self.k = 5

    def test_laplacian_operator(self):
        operator = RegularizedLaplacian(self.adjacency)
        x = operator.T.dot(np.ones((self.n, 3)))
        self.assertEqual(x.shape, (self.n, 3))

    def test_laplacian(self):
        # regular Laplacian
        spectral = LaplacianEmbedding(self.k, normalized=False)
        embedding = spectral.fit_transform(self.adjacency)
        self.assertAlmostEqual(np.linalg.norm(embedding.mean(axis=0)), 0)

        spectral = LaplacianEmbedding(self.k, regularization=0)
        with self.assertRaises(ValueError):
            spectral.fit(test_bigraph())
        with self.assertRaises(ValueError):
            spectral.fit(test_digraph())
        with self.assertRaises(ValueError):
            spectral.fit(test_graph_disconnect())

        with self.assertWarns(Warning):
            n = self.k - 1
            spectral.fit_transform(np.ones((n, n)))

    def test_spectral(self):
        spectral = Spectral(self.k)
        embedding = spectral.fit_transform(self.adjacency)
        weights = self.adjacency.dot(np.ones(self.n))
        #self.assertAlmostEqual(np.linalg.norm(embedding.T.dot(weights)), 0)
        #error = np.abs(spectral.predict(self.adjacency[:4]) - embedding[:4]).sum()
        #self.assertAlmostEqual(error, 0)

    def test_regularization(self):
        adjacency = test_graph_disconnect()
        n = adjacency.shape[0]
        spectral = Spectral(regularization=1.)
        spectral.fit(adjacency)
        spectral.predict(np.random.rand(n))
