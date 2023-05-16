#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for random projection"""
import unittest

from sknetwork.data.test_graphs import test_graph, test_bigraph, test_digraph, test_disconnected_graph
from sknetwork.embedding import RandomProjection


class TestEmbeddings(unittest.TestCase):

    def test_random_projection(self):
        for algo in [RandomProjection(), RandomProjection(random_walk=True)]:
            adjacency = test_graph()
            embedding = algo.fit_transform(adjacency)
            self.assertEqual(embedding.shape[1], 2)
            embedding = algo.fit_transform(adjacency, force_bipartite=True)
            self.assertEqual(embedding.shape[1], 2)
            adjacency = test_digraph()
            embedding = algo.fit_transform(adjacency)
            self.assertEqual(embedding.shape[1], 2)
            adjacency = test_disconnected_graph()
            embedding = algo.fit_transform(adjacency)
            self.assertEqual(embedding.shape[1], 2)
            biadjacency = test_bigraph()
            embedding = algo.fit_transform(biadjacency)
            self.assertEqual(embedding.shape[1], 2)
            self.assertEqual(algo.embedding_col_.shape[1], 2)
