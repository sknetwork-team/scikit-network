#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for embeddings metrics"""

import unittest

import numpy as np

from sknetwork.linalg.randomized_methods import randomized_svd, randomized_eig
from sknetwork.linalg.sparse_lowrank import SparseLR
from sknetwork.data import house, star_wars


class TestSparseLowRank(unittest.TestCase):

    def setUp(self):
        """Simple regularized adjacency and biadjacency for tests."""
        self.undirected = SparseLR(house(), [(np.ones(5), np.ones(5))])
        self.bipartite = SparseLR(star_wars(), [(np.ones(4), np.ones(3))])

    def test_init(self):
        with self.assertRaises(ValueError):
            SparseLR(house(), [(np.ones(5), np.ones(4))])
        with self.assertRaises(ValueError):
            SparseLR(house(), [(np.ones(4), np.ones(5))])

    def test_addition(self):
        addition = self.undirected + self.undirected
        expected = SparseLR(2 * house(), [(np.ones(5), 2 * np.ones(5))])
        err = (addition.sparse_mat - expected.sparse_mat).count_nonzero()
        self.assertEqual(err, 0)
        x = np.random.rand(5)
        self.assertAlmostEqual(np.linalg.norm(addition.dot(x) - expected.dot(x)), 0)

    def test_operations(self):
        adjacency = self.undirected.sparse_mat
        slr = -self.undirected
        slr += adjacency
        slr -= adjacency
        slr.left_sparse_dot(adjacency)
        slr.right_sparse_dot(adjacency)
        slr.astype(np.float)

    def test_product(self):
        prod = self.undirected.dot(np.ones(5))
        self.assertEqual(prod.shape, (5,))
        prod = self.bipartite.dot(np.ones(3))
        self.assertEqual(np.linalg.norm(prod - np.array([5., 4., 6., 5.])), 0.)
        prod = self.bipartite.dot(0.5 * np.ones(3))
        self.assertEqual(np.linalg.norm(prod - np.array([2.5, 2., 3., 2.5])), 0.)

    def test_transposition(self):
        transposed = self.undirected.T
        error = (self.undirected.sparse_mat - transposed.sparse_mat).data
        self.assertEqual(abs(error).sum(), 0.)
        transposed = self.bipartite.T
        x, y = transposed.low_rank_tuples[0]
        self.assertTrue((x == np.ones(3)).all())
        self.assertTrue((y == np.ones(4)).all())

    def test_decomposition(self):
        eigenvalues, eigenvectors = randomized_eig(self.undirected, n_components=2, which='LM')
        self.assertEqual(eigenvalues.shape, (2,))
        self.assertEqual(eigenvectors.shape, (5, 2))

        eigenvalues, eigenvectors = randomized_eig(self.undirected, n_components=2, which='SM')
        self.assertEqual(eigenvalues.shape, (2,))
        self.assertEqual(eigenvectors.shape, (5, 2))

        left_sv, sv, right_sv = randomized_svd(self.bipartite, n_components=2)
        self.assertEqual(left_sv.shape, (4, 2))
        self.assertEqual(sv.shape, (2,))
        self.assertEqual(right_sv.shape, (2, 3))
