#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for spectral embedding"""

import unittest

import numpy as np

from sknetwork.embedding import Spectral
from sknetwork.embedding.spectral import LaplacianOperator
from sknetwork.data.test_graphs import test_graph, test_bigraph, test_digraph, test_graph_disconnect


class TestEmbeddings(unittest.TestCase):

    def setUp(self) -> None:
        """Simple case for testing"""
        self.adjacency = test_graph()
        self.n = self.adjacency.shape[0]
        self.k = 5

    def test_laplacian_operator(self):
        operator = LaplacianOperator(self.adjacency)
        x = operator.T.dot(np.ones((self.n, 3)))
        self.assertEqual(x.shape, (self.n, 3))

    def test_regular(self):
        # regular Laplacian
        spectral = Spectral(self.k, normalized_laplacian=False, barycenter=False, normalized=False)
        embedding = spectral.fit_transform(self.adjacency)
        self.assertAlmostEqual(np.linalg.norm(embedding.mean(axis=0)), 0)
        error = np.abs(spectral.predict(self.adjacency[1]) - embedding[1]).sum()
        self.assertAlmostEqual(error, 0)

        with self.assertRaises(ValueError):
            spectral = Spectral(self.k, normalized_laplacian=False, regularization=0, equalize=True)
            spectral.fit(test_bigraph())
            spectral.fit(test_digraph())
            spectral.fit(test_graph_disconnect())

        with self.assertWarns(Warning):
            n = self.k - 1
            spectral.fit_transform(np.ones((n, n)))

    def test_normalized(self):
        # normalized Laplacian
        spectral = Spectral(self.k, barycenter=False, normalized=False)
        embedding = spectral.fit_transform(self.adjacency)
        weights = self.adjacency.dot(np.ones(self.n)) + self.n * spectral.regularization_
        self.assertAlmostEqual(np.linalg.norm(embedding.T.dot(weights)), 0)
        error = np.abs(spectral.predict(self.adjacency[:4]) - embedding[:4]).sum()
        # self.assertAlmostEqual(error, 0)

    def test_solvers(self):
        # solver
        spectral = Spectral(self.k, solver='lanczos')
        embedding = spectral.fit_transform(self.adjacency)
        self.assertEqual(embedding.shape, (self.n, self.k))
        spectral = Spectral(self.k, solver='halko')
        embedding = spectral.fit_transform(self.adjacency)
        self.assertEqual(embedding.shape, (self.n, self.k))

    def test_equalize(self):
        spectral = Spectral(self.k, equalize=True)
        spectral.fit(self.adjacency)
        spectral.predict(np.ones(self.n))
