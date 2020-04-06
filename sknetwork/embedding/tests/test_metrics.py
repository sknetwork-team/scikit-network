#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for embeddings metrics"""

import unittest

from sknetwork.embedding import GSVD
from sknetwork.embedding.metrics import cosine_modularity
from sknetwork.data.test_graphs import test_bigraph


class TestClusteringMetrics(unittest.TestCase):

    def test_bipartite(self):
        biadjacency = test_bigraph()
        method = GSVD(3, solver='lanczos')

        embedding = method.fit_transform(biadjacency)
        embedding_col = method.embedding_col_
        fit, div, modularity = cosine_modularity(biadjacency, embedding, embedding_col, return_all=True)
        self.assertAlmostEqual(modularity, fit - div)
