#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for random projection"""
import unittest

from sknetwork.data.test_graphs import test_graph, test_bigraph, test_digraph, test_graph_disconnect
from sknetwork.embedding import BiRandomProjection, RandomProjection


class TestEmbeddings(unittest.TestCase):

    def test_random_projection(self):
        for algo in [RandomProjection(), RandomProjection(random_walk=True)]:
            adjacency = test_graph()
            embedding = algo.fit_transform(adjacency)
            self.assertEqual(embedding.shape[1], 2)
            adjacency = test_digraph()
            embedding = algo.fit_transform(adjacency)
            self.assertEqual(embedding.shape[1], 2)
            adjacency = test_graph_disconnect()
            embedding = algo.fit_transform(adjacency)
            self.assertEqual(embedding.shape[1], 2)

    def test_birandom_projection(self):
        for algo in [BiRandomProjection(), BiRandomProjection(random_walk=True)]:
            biadjacency = test_bigraph()
            embedding = algo.fit_transform(biadjacency)
            self.assertEqual(embedding.shape[1], 2)
            self.assertEqual(algo.embedding_col_.shape[1], 2)
