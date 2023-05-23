#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for spectral embedding."""

import unittest

from sknetwork.data.test_graphs import *
from sknetwork.embedding import Spectral
from sknetwork.utils.check import is_weakly_connected
from sknetwork.utils.format import bipartite2undirected


class TestEmbeddings(unittest.TestCase):

    def test_undirected(self):
        for adjacency in [test_graph(), test_disconnected_graph()]:
            n = adjacency.shape[0]
            # random walk
            spectral = Spectral(3, normalized=False)
            embedding = spectral.fit_transform(adjacency)
            weights = adjacency.dot(np.ones(n))
            if not is_weakly_connected(adjacency):
                weights += 1
            self.assertAlmostEqual(np.linalg.norm(embedding.T.dot(weights)), 0)
            # Laplacian
            spectral = Spectral(3, decomposition='laplacian', normalized=False)
            embedding = spectral.fit_transform(adjacency)
            self.assertAlmostEqual(np.linalg.norm(embedding.sum(axis=0)), 0)

    def test_directed(self):
        for adjacency in [test_digraph(), test_digraph().astype(bool)]:
            # random walk
            spectral = Spectral(3, normalized=False)
            embedding = spectral.fit_transform(adjacency)
            self.assertAlmostEqual(embedding.shape[0], adjacency.shape[0])
            # Laplacian
            spectral = Spectral(3, decomposition='laplacian', normalized=False)
            spectral.fit(adjacency)
            self.assertAlmostEqual(np.linalg.norm(spectral.eigenvectors_.sum(axis=0)), 0)

    def test_regularization(self):
        for adjacency in [test_graph(), test_disconnected_graph()]:
            n = adjacency.shape[0]
            # random walk
            regularization = 0.1
            spectral = Spectral(3, regularization=regularization, normalized=False)
            embedding = spectral.fit_transform(adjacency)
            weights = adjacency.dot(np.ones(n)) + regularization
            self.assertAlmostEqual(np.linalg.norm(embedding.T.dot(weights)), 0)
            # Laplacian
            spectral = Spectral(3, decomposition='laplacian', regularization=1, normalized=False)
            embedding = spectral.fit_transform(adjacency)
            self.assertAlmostEqual(np.linalg.norm(embedding.sum(axis=0)), 0)
            # without regularization
            spectral = Spectral(3, decomposition='laplacian', regularization=-1, normalized=False)
            embedding = spectral.fit_transform(adjacency)
            self.assertAlmostEqual(np.linalg.norm(embedding.sum(axis=0)), 0)

    def test_bipartite(self):
        for biadjacency in [test_digraph(), test_bigraph(), test_bigraph_disconnect()]:
            n_row, n_col = biadjacency.shape
            adjacency = bipartite2undirected(biadjacency)
            # random walk
            spectral = Spectral(3, normalized=False)
            spectral.fit(biadjacency)
            embedding_full = np.vstack([spectral.embedding_row_, spectral.embedding_col_])
            weights = adjacency.dot(np.ones(n_row + n_col))
            if not is_weakly_connected(adjacency):
                weights += 1
            self.assertAlmostEqual(np.linalg.norm(embedding_full.T.dot(weights)), 0)
            # Laplacian
            spectral = Spectral(3, decomposition='laplacian', normalized=False)
            spectral.fit(biadjacency)
            embedding_full = np.vstack([spectral.embedding_row_, spectral.embedding_col_])
            self.assertAlmostEqual(np.linalg.norm(embedding_full.sum(axis=0)), 0)

    def test_normalization(self):
        for adjacency in [test_graph(), test_disconnected_graph()]:
            spectral = Spectral(3)
            embedding = spectral.fit_transform(adjacency)
            self.assertAlmostEqual(np.linalg.norm(np.linalg.norm(embedding, axis=1) - np.ones(adjacency.shape[0])), 0)
