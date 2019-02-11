#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for embeddings"""

import unittest
from sknetwork.embedding.gsvd import GSVDEmbedding
from sknetwork.embedding.spectral import SpectralEmbedding
from scipy import sparse
import numpy as np


class TestClusteringMetrics(unittest.TestCase):

    def setUp(self):
        self.graph = sparse.csr_matrix(np.array([[0, 1, 1, 1],
                                                 [1, 0, 0, 0],
                                                 [1, 0, 0, 1],
                                                 [1, 0, 1, 0]]))
        self.bipartite = sparse.csr_matrix(np.array([[1, 0, 1],
                                                     [1, 0, 0],
                                                     [1, 1, 1],
                                                     [0, 1, 1]]))

    def test_gsvd(self):
        gsvd = GSVDEmbedding(2)
        gsvd.fit(self.graph)
        self.assertTrue(gsvd.embedding_.shape == (4, 2))
        self.assertTrue(gsvd.features_.shape == (4, 2))
        self.assertTrue(type(gsvd.singular_values_) == np.ndarray and len(gsvd.singular_values_) == 2)

        gsvd.fit(self.bipartite)
        self.assertTrue(gsvd.embedding_.shape == (4, 2))
        self.assertTrue(gsvd.features_.shape == (3, 2))
        self.assertTrue(type(gsvd.singular_values_) == np.ndarray and len(gsvd.singular_values_) == 2)

    def test_spectral(self):
        sp = SpectralEmbedding(2)
        sp.fit(self.graph)
        self.assertTrue(sp.embedding_.shape == (4, 2))
        self.assertTrue(type(sp.eigenvalues_) == np.ndarray and len(sp.eigenvalues_) == 2)
