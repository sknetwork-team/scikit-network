#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for embeddings"""

import unittest
from sknetwork.embedding import GSVD
from sknetwork.embedding import Spectral
from scipy import sparse
import numpy as np


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
        self.assertEqual(spectral.features_.shape, (3, 2))
        self.assertTrue(type(spectral.eigenvalues_) == np.ndarray and len(spectral.eigenvalues_) == 2)

    def test_gsvd(self):
        gsvd = GSVD(2)
        gsvd.fit(self.adjacency)
        self.assertTrue(gsvd.embedding_.shape == (4, 2))
        self.assertTrue(gsvd.features_.shape == (4, 2))
        self.assertTrue(type(gsvd.singular_values_) == np.ndarray and len(gsvd.singular_values_) == 2)

        gsvd.fit(self.biadjacency)
        self.assertTrue(gsvd.embedding_.shape == (4, 1))
        self.assertTrue(gsvd.features_.shape == (3, 1))
        self.assertTrue(type(gsvd.singular_values_) == np.ndarray and len(gsvd.singular_values_) == 1)
