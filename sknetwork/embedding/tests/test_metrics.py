#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for embeddings metrics"""

import unittest

from sknetwork.data.test_graphs import test_bigraph, test_graph
from sknetwork.embedding import Spectral
from sknetwork.embedding.metrics import get_cosine_similarity


class TestClusteringMetrics(unittest.TestCase):

    def test_cosine(self):
        biadjacency = test_bigraph()
        method = Spectral(3)

        embedding = method.fit_transform(biadjacency)
        embedding_col = method.embedding_col_
        similarity = get_cosine_similarity(biadjacency, embedding, embedding_col)
        self.assertAlmostEqual(similarity, 0.56, 2)

        adjacency = test_graph()
        embedding = method.fit_transform(adjacency)
        similarity = get_cosine_similarity(adjacency, embedding)
        self.assertAlmostEqual(similarity, 0.32, 2)

        with self.assertRaises(ValueError):
            get_cosine_similarity(biadjacency, embedding)
