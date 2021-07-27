#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for embeddings"""

import unittest

from sknetwork.data.test_graphs import *
from sknetwork.embedding import Spectral, SVD, GSVD, Spring


class TestEmbeddings(unittest.TestCase):

    def setUp(self):
        """Algorithms by input types."""
        self.methods = [Spectral(), GSVD(), SVD()]
        self.bimethods = [GSVD(), SVD()]

    def test_undirected(self):
        adjacency = test_graph()
        n = adjacency.shape[0]

        method = Spring()
        embedding = method.fit_transform(adjacency)
        self.assertEqual(embedding.shape, (n, 2))
        pred1 = method.predict(adjacency[0])
        pred2 = method.predict(adjacency[0].toarray())
        self.assertEqual(pred1.shape, (2,))
        self.assertAlmostEqual(np.linalg.norm(pred1 - pred2), 0)

        pred1 = method.predict(adjacency)
        pred2 = method.predict(adjacency.toarray())
        self.assertTupleEqual(pred1.shape, (n, 2))
        self.assertAlmostEqual(np.linalg.norm(pred1 - pred2), 0)

    def test_bimethods(self):

        for adjacency in [test_digraph(), test_bigraph()]:
            n_row, n_col = adjacency.shape

            for method in self.bimethods:
                method.fit(adjacency)

                self.assertEqual(method.embedding_.shape, (n_row, 2))
                self.assertEqual(method.embedding_row_.shape, (n_row, 2))
                self.assertEqual(method.embedding_col_.shape, (n_col, 2))

                ref = method.embedding_[0]
                pred1 = method.predict(adjacency[0])
                pred2 = method.predict(adjacency[0].toarray())

                self.assertEqual(pred1.shape, (2,))
                self.assertAlmostEqual(np.linalg.norm(pred1 - pred2), 0)
                self.assertAlmostEqual(np.linalg.norm(pred1 - ref), 0)

                ref = method.embedding_
                pred1 = method.predict(adjacency)
                pred2 = method.predict(adjacency.toarray())

                self.assertTupleEqual(pred1.shape, (n_row, 2))
                self.assertAlmostEqual(np.linalg.norm(pred1 - pred2), 0)
                self.assertAlmostEqual(np.linalg.norm(pred1 - ref), 0)

    def test_disconnected(self):
        n = 10
        adjacency = np.eye(n)
        for method in self.methods:
            embedding = method.fit_transform(adjacency)
            self.assertEqual(embedding.shape, (n, 2))

    def test_regularization(self):
        adjacency = test_graph()
        method = Spectral()
        self.assertEqual(method._get_regularization(-1, adjacency), 0)
