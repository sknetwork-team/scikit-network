#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for embeddings"""

import unittest

from sknetwork.embedding import Spectral, BiSpectral, SVD, GSVD
from sknetwork.data.test_graphs import *


# noinspection PyMissingOrEmptyDocstring
class TestEmbeddings(unittest.TestCase):

    def setUp(self):
        self.methods = [Spectral(), SVD(), GSVD()]
        self.bimethods = [BiSpectral(), SVD(), GSVD()]

    def test_undirected(self):
        adjacency = Simple().adjacency
        n = adjacency.shape[0]
        for method in self.methods:
            embedding = method.fit_transform(adjacency)
            self.assertEqual(embedding.shape, (n, 2))
            error = np.abs(method.predict(adjacency[0]) - method.embedding_[0]).sum()
            #self.assertAlmostEqual(error, 0)
            error = np.abs(method.predict(adjacency) - method.embedding_).sum()
            #self.assertAlmostEqual(error, 0)

    def test_directed(self):
        adjacency = DiSimple().adjacency
        n = adjacency.shape[0]
        for method in self.bimethods:
            method.fit(adjacency)
            self.assertEqual(method.embedding_.shape, (n, 2))
            self.assertEqual(method.embedding_row_.shape, (n, 2))
            self.assertEqual(method.embedding_col_.shape, (n, 2))
            error = np.abs(method.predict(adjacency[0]) - method.embedding_[0]).sum()
            #self.assertAlmostEqual(error, 0)
            error = np.abs(method.predict(adjacency) - method.embedding_).sum()
            #self.assertAlmostEqual(error, 0)

    def test_bipartite(self):
        biadjacency = BiSimple().biadjacency
        n1, n2 = biadjacency.shape
        for method in self.bimethods:
            method.fit(biadjacency)
            self.assertEqual(method.embedding_.shape, (n1, 2))
            self.assertEqual(method.embedding_row_.shape, (n1, 2))
            self.assertEqual(method.embedding_col_.shape, (n2, 2))
            error = np.abs(method.predict(biadjacency[0]) - method.embedding_[0]).sum()
            #self.assertAlmostEqual(error, 0)
            error = np.abs(method.predict(biadjacency) - method.embedding_).sum()
            #self.assertAlmostEqual(error, 0)

    def test_disconnected(self):
        adjacency = np.eye(10)
        for method in self.methods:
            embedding = method.fit_transform(adjacency)
            self.assertEqual(embedding.shape, (10, 2))

    def test_spectral_options(self):
        adjacency = Simple().adjacency
        n = adjacency.shape[0]
        k = 5

        # regular Laplacian
        spectral = Spectral(k, normalized_laplacian=False, barycenter=False, normalize=False)
        embedding = spectral.fit_transform(adjacency)
        self.assertAlmostEqual(np.linalg.norm(embedding.mean(axis=0)), 0)
        error = np.abs(spectral.predict(adjacency[1]) - embedding[1]).sum()
        #self.assertAlmostEqual(error, 0)

        # normalized Laplacian
        spectral = Spectral(k,  barycenter=False, normalize=False)
        embedding = spectral.fit_transform(adjacency)
        weights = adjacency.dot(np.ones(n)) + n * spectral.regularization_
        self.assertAlmostEqual(np.linalg.norm(embedding.T.dot(weights)), 0)
        error = np.abs(spectral.predict(adjacency[:4]) - embedding[:4]).sum()
        #self.assertAlmostEqual(error, 0)

        # solver
        spectral = Spectral(k, solver='lanczos')
        embedding = spectral.fit_transform(adjacency)
        self.assertEqual(embedding.shape, (n, k))
        spectral = Spectral(k, solver='halko')
        embedding = spectral.fit_transform(adjacency)
        self.assertEqual(embedding.shape, (n, k))

    # noinspection PyTypeChecker
    def test_input(self):
        for method in self.methods:
            with self.assertRaises(TypeError):
                method.fit(sparse.identity(5))
