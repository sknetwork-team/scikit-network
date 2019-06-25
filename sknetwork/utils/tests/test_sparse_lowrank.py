#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for embeddings metrics"""

import unittest
import numpy as np
from scipy import sparse
from sknetwork.toy_graphs import house_graph, star_wars_villains_graph
from sknetwork.utils.sparse_lowrank import SparseLR
from sknetwork.utils import randomized_svd, randomized_eig


class TestSparseLowRank(unittest.TestCase):

    def setUp(self):
        self.undirected = SparseLR(house_graph(), [(np.ones(5), np.ones(5))])
        self.bipartite = SparseLR(star_wars_villains_graph(), [(np.ones(4), np.ones(3))])

    def test_addition(self):
        addition = self.undirected + self.undirected
        expected = SparseLR(2 * house_graph(), [(np.ones(5), 2 * np.ones(5))])
        err: sparse.csr_matrix = (addition.sparse_mat != expected.sparse_mat)
        self.assertEqual(err.nnz, 0)
        random_vector = np.random.rand(5)
        self.assertAlmostEqual(np.linalg.norm(addition.dot(random_vector) - expected.dot(random_vector)), 0)

    def test_product(self):
        prod = self.undirected.dot(np.ones(5))
        self.assertEqual(prod.shape, (5,))
        prod = self.bipartite.dot(np.ones(3))
        self.assertEqual(np.linalg.norm(prod - np.array([5., 4., 6., 5.])), 0.)

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
