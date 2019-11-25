#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for embeddings metrics"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.embedding.metrics import cosine_modularity
from sknetwork.data import star_wars_villains


class TestClusteringMetrics(unittest.TestCase):

    def test_cosine_modularity(self):
        self.graph = sparse.csr_matrix(np.array([[0, 1, 1, 1],
                                                 [1, 0, 0, 0],
                                                 [1, 0, 0, 1],
                                                 [1, 0, 1, 0]]))
        self.bipartite: sparse.csr_matrix = star_wars_villains()
        self.embedding = np.ones((4, 2))
        self.features = np.ones((3, 2))

        self.assertAlmostEqual(cosine_modularity(self.graph, self.embedding), 0.)
        fit, div, modularity = cosine_modularity(self.graph, self.embedding, return_all=True)
        self.assertAlmostEqual(fit, 1.)
        self.assertAlmostEqual(div, 1.)

        self.assertAlmostEqual(cosine_modularity(self.bipartite, self.embedding, self.features), 0.)
        fit, div, modularity = cosine_modularity(self.bipartite, self.embedding, self.features, return_all=True)
        self.assertAlmostEqual(fit, 1.)
        self.assertAlmostEqual(div, 1.)
