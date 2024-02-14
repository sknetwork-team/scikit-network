#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for Louvain embedding"""
import unittest

import numpy as np

from sknetwork.data.test_graphs import test_graph, test_bigraph
from sknetwork.embedding import LouvainEmbedding


class TestLouvainEmbedding(unittest.TestCase):

    def test_predict(self):
        adjacency = test_graph()
        louvain = LouvainEmbedding()
        louvain.fit(test_graph())
        self.assertEqual(louvain.embedding_.shape[0], 10)
        louvain.fit(adjacency, force_bipartite=True)
        self.assertEqual(louvain.embedding_.shape[0], 10)

        for method in ['remove', 'merge', 'keep']:
            louvain = LouvainEmbedding(isolated_nodes=method)
            embedding = louvain.fit_transform(adjacency)
            self.assertEqual(embedding.shape[0], adjacency.shape[0])



