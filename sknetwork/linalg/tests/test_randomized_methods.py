#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for randomized matrix factorization"""

import unittest

from sknetwork.linalg.randomized_methods import *
from sknetwork.linalg.sparse_lowrank import SparseLR


class TestClusteringMetrics(unittest.TestCase):

    def test_dot(self):
        n = 5
        x = np.random.randn(n)
        mat = sparse.eye(n, format='csr')
        slr = SparseLR(mat, [(np.zeros(n), np.zeros(n))])

        y1 = safe_sparse_dot(x, slr)
        y2 = safe_sparse_dot(slr, x)
        self.assertAlmostEqual(np.linalg.norm(y1 - y2), 0)

        with self.assertRaises(NotImplementedError):
            safe_sparse_dot(slr, slr)

        y1 = safe_sparse_dot(slr, mat).dot(x)
        y2 = safe_sparse_dot(mat, slr).dot(x)
        self.assertAlmostEqual(np.linalg.norm(y1 - y2), 0)

    def test_range_finder(self):
        x = np.random.randn(3, 7)
        y1 = randomized_range_finder(x, size=2, n_iter=2, power_iteration_normalizer='auto')
        y2 = randomized_range_finder(x, size=2, n_iter=2, power_iteration_normalizer='LU')
        y3 = randomized_range_finder(x, size=2, n_iter=2, power_iteration_normalizer='QR')

        self.assertEqual(y1.shape, (3, 2))
        self.assertEqual(y2.shape, (3, 2))
        self.assertEqual(y3.shape, (3, 2))

    def test_eig(self):
        eigenvalues, eigenvectors = randomized_eig(sparse.identity(5, format='csr'), 2)
        self.assertTrue(np.allclose(eigenvalues, np.array([1., 1.])))
        self.assertTrue(eigenvectors.shape == (5, 2))

        eigenvalues, eigenvectors = randomized_eig(sparse.identity(5, format='csr'), 2, one_pass=True)
        self.assertTrue(np.allclose(eigenvalues, np.array([1., 1.])))
        self.assertTrue(eigenvectors.shape == (5, 2))

    def test_svd(self):
        self.sym_matrix = sparse.csr_matrix(np.array([[0, 1, 1, 1],
                                                      [1, 0, 0, 0],
                                                      [1, 0, 0, 1],
                                                      [1, 0, 1, 0]]))
        left_singular_vectors, singular_values, _ = randomized_svd(sparse.identity(5, format='csr'), 2)
        self.assertTrue(np.allclose(singular_values, np.array([1., 1.])))
        self.assertTrue(left_singular_vectors.shape == (5, 2))

        _, singular_values, _ = randomized_svd(self.sym_matrix, 2)
        self.assertTrue(np.allclose(singular_values, np.array([2.17, 1.48]), 1e-2))
