#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for embeddings metrics"""

import unittest

from sknetwork.embedding import GSVD
from sknetwork.embedding.metrics import cosine_modularity
from sknetwork.data.test_graphs import *


class TestClusteringMetrics(unittest.TestCase):

    def setUp(self):
        self.method = GSVD(3, solver='lanczos')

    def test_bipartite(self):
        biadjacency = test_bigraph()
        embedding = self.method.fit_transform(biadjacency)
        embedding_col = self.method.embedding_col_
        fit, div, modularity = cosine_modularity(biadjacency, embedding, embedding_col, return_all=True)
        self.assertAlmostEqual(modularity, 0.302, 2)
        self.assertAlmostEqual(fit, 1., 2)
        self.assertAlmostEqual(div, 0.697, 2)
