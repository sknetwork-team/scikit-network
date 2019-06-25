#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for embeddings metrics"""

import unittest
import numpy as np
from scipy import sparse
from sknetwork.embedding.metrics import dot_modularity, hscore, linear_fit
from sknetwork.toy_graphs import star_wars_villains_graph


class TestClusteringMetrics(unittest.TestCase):

    def setUp(self):
        self.graph = sparse.csr_matrix(np.array([[0, 1, 1, 1],
                                                 [1, 0, 0, 0],
                                                 [1, 0, 0, 1],
                                                 [1, 0, 1, 0]]))
        self.bipartite = star_wars_villains_graph()
        self.embedding = np.ones((4, 2))
        self.features = np.ones((3, 2))

    def test_dot_modularity(self):
        self.assertAlmostEqual(dot_modularity(self.graph, self.embedding), 0.)
        fit, diversity = dot_modularity(self.graph, self.embedding, return_all=True)
        self.assertAlmostEqual(fit, 1.)
        self.assertAlmostEqual(diversity, 1.)

        self.assertAlmostEqual(dot_modularity(self.bipartite, self.embedding, self.features), 0.)
        fit, diversity = dot_modularity(self.bipartite, self.embedding, self.features, return_all=True)
        self.assertAlmostEqual(fit, 1.)
        self.assertAlmostEqual(diversity, 1.)

    def test_hscore(self):
        self.assertAlmostEqual(hscore(self.graph, self.embedding), 0.)
        fit, diversity = hscore(self.graph, self.embedding, return_all=True)
        self.assertAlmostEqual(fit, 1.)
        self.assertAlmostEqual(diversity, 0.)

    def test_linear_fit(self):
        embedding = self.embedding / np.sqrt(2)
        fit, div = linear_fit(self.graph, embedding)
        self.assertAlmostEqual(fit, 1.)
        self.assertAlmostEqual(div, 1.)
        fit, div = linear_fit(self.graph, embedding, damping=0.15)
        self.assertAlmostEqual(fit, 1.)
        self.assertAlmostEqual(div, 1.)
        fit, div = linear_fit(self.graph, embedding, order=3)
        self.assertAlmostEqual(fit, 1.)
        self.assertAlmostEqual(div, 1.)
