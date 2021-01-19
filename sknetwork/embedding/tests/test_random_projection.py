#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for random projection"""
import unittest

from sknetwork.data.test_graphs import test_graph, test_bigraph, test_digraph, test_graph_disconnect
from sknetwork.embedding import BiRandomProjection, RandomProjection


class TestEmbeddings(unittest.TestCase):

    def test_random_projection(self):
        random_projection = RandomProjection()
        adjacency = test_graph()
        embedding = random_projection.fit_transform(adjacency)
        self.assertEqual(embedding.shape[1], 2)
        adjacency = test_digraph()
        embedding = random_projection.fit_transform(adjacency)
        self.assertEqual(embedding.shape[1], 2)
        adjacency = test_graph_disconnect()
        embedding = random_projection.fit_transform(adjacency)
        self.assertEqual(embedding.shape[1], 2)

    def test_birandom_projection(self):
        birandom_projection = BiRandomProjection()
        biadjacency = test_bigraph()
        embedding = birandom_projection.fit_transform(biadjacency)
        self.assertEqual(embedding.shape[1], 2)
        self.assertEqual(birandom_projection.embedding_col_.shape[1], 2)
