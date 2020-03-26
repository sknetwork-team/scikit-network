#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for embeddings"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.embedding import Spectral, BiSpectral, SVD, GSVD
from sknetwork.data import karate_club, painters, movie_actor


def barycenter_norm(adjacency, spectral: Spectral) -> float:
    """Barycenter of the embedding with respect to adjacency weights.

    Parameters
    ----------
    adjacency
    spectral

    Returns
    -------
    barycenter

    """
    if spectral.normalized_laplacian:
        weights = adjacency.dot(np.ones(adjacency.shape[1]))
        if spectral.regularization:
            weights += spectral.regularization * adjacency.shape[1]
        return np.linalg.norm(spectral.embedding_.T.dot(weights)) / weights.sum()
    else:
        return np.linalg.norm(spectral.embedding_.mean(axis=0))


# noinspection PyMissingOrEmptyDocstring
class TestEmbeddings(unittest.TestCase):

    def setUp(self):
        self.methods = [Spectral(), SVD(), GSVD()]
        self.bimethods = [BiSpectral(), SVD(), GSVD()]

    def test_undirected(self):
        adjacency = karate_club()
        n = adjacency.shape[0]
        for method in self.methods:
            embedding = method.fit_transform(adjacency)
            self.assertEqual(embedding.shape, (n, 2))
            error = np.abs(method.predict(adjacency[0]) - method.embedding_[0]).sum()
            self.assertAlmostEqual(error, 0)
            error = np.abs(method.predict(adjacency) - method.embedding_).sum()
            self.assertAlmostEqual(error, 0)

    def test_directed(self):
        adjacency = painters()
        n = adjacency.shape[0]
        for method in self.bimethods:
            method.fit(adjacency)
            self.assertEqual(method.embedding_.shape, (n, 2))
            self.assertEqual(method.embedding_row_.shape, (n, 2))
            self.assertEqual(method.embedding_col_.shape, (n, 2))
            error = np.abs(method.predict(adjacency[0]) - method.embedding_[0]).sum()
            self.assertAlmostEqual(error, 0)
            error = np.abs(method.predict(adjacency) - method.embedding_).sum()
            self.assertAlmostEqual(error, 0)

    def test_bipartite(self):
        biadjacency = movie_actor()
        n1, n2 = biadjacency.shape
        for method in self.bimethods:
            method.fit(biadjacency)
            self.assertEqual(method.embedding_.shape, (n1, 2))
            self.assertEqual(method.embedding_row_.shape, (n1, 2))
            self.assertEqual(method.embedding_col_.shape, (n2, 2))
            error = np.abs(method.predict(biadjacency[0]) - method.embedding_[0]).sum()
            self.assertAlmostEqual(error, 0)
            error = np.abs(method.predict(biadjacency) - method.embedding_).sum()
            self.assertAlmostEqual(error, 0)

    def test_disconnected(self):
        adjacency = np.eye(10)
        for method in self.methods:
            embedding = method.fit_transform(adjacency)
            self.assertEqual(embedding.shape, (10, 2))

    # noinspection PyTypeChecker
    def test_input(self):
        for method in self.methods:
            with self.assertRaises(TypeError):
                method.fit(sparse.identity(5))
