#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for spectral embedding."""

import unittest

from sknetwork.data.test_graphs import *
from sknetwork.embedding import Spectral
from sknetwork.topology import is_connected
from sknetwork.utils import bipartite2undirected


class TestEmbeddings(unittest.TestCase):

    def test_undirected(self):
        for adjacency in [test_graph(), test_graph_disconnect()]:
            n = adjacency.shape[0]
            # normalized Laplacian
            spectral = Spectral(3)
            embedding = spectral.fit_transform(adjacency)
            weights = adjacency.dot(np.ones(n))
            if not is_connected(adjacency):
                weights += 1
            self.assertAlmostEqual(np.linalg.norm(embedding.T.dot(weights)), 0)
            # regular Laplacian
            spectral = Spectral(3, normalized_laplacian=False)
            embedding = spectral.fit_transform(adjacency)
            self.assertAlmostEqual(np.linalg.norm(embedding.sum(axis=0)), 0)

    def test_regularization(self):
        for adjacency in [test_graph(), test_graph_disconnect()]:
            n = adjacency.shape[0]
            # normalized Laplacian
            regularization = 0.1
            spectral = Spectral(3, regularization=regularization)
            embedding = spectral.fit_transform(adjacency)
            weights = adjacency.dot(np.ones(n)) + regularization
            self.assertAlmostEqual(np.linalg.norm(embedding.T.dot(weights)), 0)
            # regular Laplacian
            spectral = Spectral(3, normalized_laplacian=False, regularization=regularization)
            embedding = spectral.fit_transform(adjacency)
            self.assertAlmostEqual(np.linalg.norm(embedding.sum(axis=0)), 0)

    def test_bipartite(self):
        for biadjacency in [test_digraph(), test_bigraph(), test_bigraph_disconnect()]:
            n_row, n_col = biadjacency.shape
            adjacency = bipartite2undirected(biadjacency)
            # normalized Laplacian
            spectral = Spectral(3)
            spectral.fit(biadjacency)
            embedding_full = np.vstack([spectral.embedding_row_, spectral.embedding_col_])
            weights = adjacency.dot(np.ones(n_row + n_col))
            if not is_connected(adjacency):
                weights += 1
            self.assertAlmostEqual(np.linalg.norm(embedding_full.T.dot(weights)), 0)
            # regular Laplacian
            spectral = Spectral(3, normalized_laplacian=False)
            spectral.fit(biadjacency)
            embedding_full = np.vstack([spectral.embedding_row_, spectral.embedding_col_])
            self.assertAlmostEqual(np.linalg.norm(embedding_full.sum(axis=0)), 0)
