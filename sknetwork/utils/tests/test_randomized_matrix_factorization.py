#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for randomized matrix factorization"""

import unittest
import numpy as np
from scipy import sparse
from sknetwork.utils.randomized_matrix_factorization import randomized_eig, randomized_svd


class TestClusteringMetrics(unittest.TestCase):

    def setUp(self):
        self.sym_matrix = sparse.csr_matrix(np.array([[0, 1, 1, 1],
                                                      [1, 0, 0, 0],
                                                      [1, 0, 0, 1],
                                                      [1, 0, 1, 0]]))

    def test_eig(self):
        eigenvalues, eigenvectors = randomized_eig(sparse.identity(5, format='csr'), 2)
        self.assertTrue(np.allclose(eigenvalues, np.array([1., 1.])))
        self.assertTrue(eigenvectors.shape == (5, 2))

    def test_svd(self):
        left_singular_vectors, singular_values, _ = randomized_svd(sparse.identity(5, format='csr'), 2)
        self.assertTrue(np.allclose(singular_values, np.array([1., 1.])))
        self.assertTrue(left_singular_vectors.shape == (5, 2))

        _, singular_values, _ = randomized_svd(self.sym_matrix, 2)
        self.assertTrue(np.allclose(singular_values, np.array([2.17, 1.48]), 1e-2))
