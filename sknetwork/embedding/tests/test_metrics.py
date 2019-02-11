#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for embeddings metrics"""

import unittest
import numpy as np
from scipy import sparse
from sknetwork.embedding.metrics import dot_modularity


class TestClusteringMetrics(unittest.TestCase):

    def setUp(self):
        self.graph = sparse.csr_matrix(np.array([[0, 1, 1, 1],
                                                 [1, 0, 0, 0],
                                                 [1, 0, 0, 1],
                                                 [1, 0, 1, 0]]))
        self.embedding = np.ones((4, 2))

    def test_dot_modularity(self):
        self.assertAlmostEqual(dot_modularity(self.graph, self.embedding), 0.)
        fit, diversity = dot_modularity(self.graph, self.embedding, return_all=True)
        self.assertAlmostEqual(fit, 1.)
        self.assertAlmostEqual(diversity, 1.)

