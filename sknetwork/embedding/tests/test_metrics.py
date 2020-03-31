#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for embeddings metrics"""

import unittest

from sknetwork.embedding import GSVD
from sknetwork.embedding.metrics import cosine_modularity
from sknetwork.data.basic import *


class TestClusteringMetrics(unittest.TestCase):

    def setUp(self):
        self.method = GSVD(3, solver='lanczos')

    def test_undirected(self):
        adjacency = SmallDisconnected().adjacency
        embedding = self.method.fit_transform(adjacency)
        fit, div, modularity = cosine_modularity(adjacency, embedding, return_all=True)
        self.assertAlmostEqual(modularity, 0.07, 2)
        self.assertAlmostEqual(fit, 0.894, 2)
        self.assertAlmostEqual(div, 0.826, 2)

    def test_directed(self):
        adjacency = DiSmall().adjacency
        embedding = self.method.fit_transform(adjacency)
        fit, div, modularity = cosine_modularity(adjacency, embedding, return_all=True)
        self.assertAlmostEqual(modularity, 0.16, 1)
        self.assertAlmostEqual(fit, 0.71, 1)
        self.assertAlmostEqual(div, 0.55, 1)

    def test_bipartite(self):
        biadjacency = BiSmall().biadjacency
        embedding = self.method.fit_transform(biadjacency)
        embedding_col = self.method.embedding_col_
        fit, div, modularity = cosine_modularity(biadjacency, embedding, embedding_col, return_all=True)
        self.assertAlmostEqual(modularity, 0.302, 2)
        self.assertAlmostEqual(fit, 1., 2)
        self.assertAlmostEqual(div, 0.697, 2)
