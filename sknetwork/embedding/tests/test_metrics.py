#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for embeddings metrics"""

import unittest

from sknetwork.data.test_graphs import test_bigraph, test_graph
from sknetwork.embedding import GSVD, LouvainEmbedding
from sknetwork.embedding.metrics import cosine_modularity


class TestClusteringMetrics(unittest.TestCase):

    def test_cosine(self):
        biadjacency = test_bigraph()
        method = GSVD(3, solver='lanczos')

        embedding = method.fit_transform(biadjacency)
        embedding_col = method.embedding_col_
        fit, div, modularity = cosine_modularity(biadjacency, embedding, embedding_col, return_all=True)
        modularity = cosine_modularity(biadjacency, embedding, embedding_col, return_all=False)
        self.assertAlmostEqual(modularity, fit - div)

        adjacency = test_graph()
        embedding = method.fit_transform(adjacency)
        fit, div, modularity = cosine_modularity(adjacency, embedding, return_all=True)
        self.assertAlmostEqual(modularity, fit - div)

        with self.assertRaises(ValueError):
            cosine_modularity(biadjacency, embedding)

        louvain = LouvainEmbedding()
        embedding = louvain.fit_transform(adjacency)
        fit, div, modularity = cosine_modularity(adjacency, embedding, return_all=True)
        self.assertAlmostEqual(modularity, fit - div)
