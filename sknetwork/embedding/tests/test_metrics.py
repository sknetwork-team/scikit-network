# -*- coding: utf-8 -*-
# test for metrics.py
#
# Copyright 2018 Scikit-network Developers.
# Copyright 2018 Nathan de Lara <ndelara@enst.fr>
#
# This file is part of Scikit-network.

import unittest
import numpy as np
from scipy import sparse
from sknetwork.embedding.metrics import cosine_modularity


class TestClusteringMetrics(unittest.TestCase):

    def setUp(self):
        self.graph = sparse.csr_matrix(np.array([[0, 1, 1, 1],
                                                 [1, 0, 0, 0],
                                                 [1, 0, 0, 1],
                                                 [1, 0, 1, 0]]))
        self.embedding = np.ones((4, 2))

    def test_cosine_modularity(self):
        self.assertAlmostEqual(cosine_modularity(self.graph, self.embedding), 0.)
