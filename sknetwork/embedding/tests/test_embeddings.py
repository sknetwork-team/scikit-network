# -*- coding: utf-8 -*-
# test for metrics.py
#
# Copyright 2018 Scikit-network Developers.
# Copyright 2018 Nathan de Lara <ndelara@enst.fr>
#
# This file is part of Scikit-network.

import unittest
from sknetwork.embedding.svd import ForwardBackwardEmbedding
from sknetwork.embedding.spectral import SpectralEmbedding
from scipy import sparse
import numpy as np


class TestClusteringMetrics(unittest.TestCase):

    def setUp(self):
        self.graph = sparse.csr_matrix(np.array([[0, 1, 1, 1],
                                                 [1, 0, 0, 0],
                                                 [1, 0, 0, 1],
                                                 [1, 0, 1, 0]]))

    def test_forwardbackward(self):
        fb = ForwardBackwardEmbedding(2)
        fb.fit(self.graph)
        self.assertTrue(fb.embedding_.shape == (4, 2))
        self.assertTrue(fb.backward_embedding_.shape == (4, 2))
        self.assertTrue(type(fb.singular_values_) == np.ndarray and len(fb.singular_values_) == 2)

    def test_spectral(self):
        sp = SpectralEmbedding(2)
        sp.fit(self.graph)
        self.assertTrue(sp.embedding_.shape == (4, 2))
        self.assertTrue(type(sp.eigenvalues_) == np.ndarray and len(sp.eigenvalues_) == 2)
