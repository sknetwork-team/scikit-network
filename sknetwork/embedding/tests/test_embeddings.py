#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for embeddings"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.embedding import SVD
from sknetwork.embedding import Spectral


class TestEmbeddings(unittest.TestCase):

    def setUp(self):
        self.adjacency = sparse.csr_matrix(np.array([[0, 1, 1, 1],
                                                     [1, 0, 0, 0],
                                                     [1, 0, 0, 1],
                                                     [1, 0, 1, 0]]))
        self.biadjacency = sparse.csr_matrix(np.array([[1, 0, 1],
                                                       [1, 0, 0],
                                                       [1, 1, 1],
                                                       [0, 1, 1]]))

    def test_spectral(self):
        spectral = Spectral(2)
        spectral.fit(self.adjacency)
        self.assertTrue(spectral.embedding_.shape == (4, 2))
        self.assertTrue(type(spectral.eigenvalues_) == np.ndarray and len(spectral.eigenvalues_) == 2)
        self.assertTrue(min(spectral.eigenvalues_ >= 0))
        self.assertTrue(max(spectral.eigenvalues_ <= 2))

        spectral.fit(self.biadjacency)
        self.assertEqual(spectral.embedding_.shape, (4, 2))
        self.assertEqual(spectral.coembedding_.shape, (3, 2))
        self.assertTrue(type(spectral.eigenvalues_) == np.ndarray and len(spectral.eigenvalues_) == 2)

    def test_svd(self):
        svd = SVD(2)
        svd.fit(self.adjacency)
        self.assertTrue(svd.embedding_.shape == (4, 2))
        self.assertTrue(svd.coembedding_.shape == (4, 2))
        self.assertTrue(type(svd.singular_values_) == np.ndarray and len(svd.singular_values_) == 2)

        svd.fit(self.biadjacency)
        self.assertTrue(svd.embedding_.shape == (4, 1))
        self.assertTrue(svd.coembedding_.shape == (3, 1))
        self.assertTrue(type(svd.singular_values_) == np.ndarray and len(svd.singular_values_) == 1)
