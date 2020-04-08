#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Louvain"""

import unittest

from sknetwork.clustering import Louvain, BiLouvain, modularity
from sknetwork.data.test_graphs import *
from sknetwork.data import karate_club


# noinspection PyMissingOrEmptyDocstring
class TestLouvainClustering(unittest.TestCase):

    def setUp(self):
        self.louvain = Louvain()
        self.bilouvain = BiLouvain()

    def test_undirected(self):
        adjacency = test_graph()
        labels = self.louvain.fit_transform(adjacency)
        self.assertEqual(labels.shape, (10,))
        self.assertAlmostEqual(modularity(adjacency, labels), 0.495, 2)

    def test_directed(self):
        adjacency = test_digraph()
        labels = self.louvain.fit_transform(adjacency)
        self.assertEqual(labels.shape, (10,))
        self.assertAlmostEqual(modularity(adjacency, labels), 0.475, 2)

    def test_bipartite(self):
        biadjacency = test_bigraph()
        n1, n2 = biadjacency.shape
        self.bilouvain.fit(biadjacency)
        labels_row = self.bilouvain.labels_row_
        labels_col = self.bilouvain.labels_col_
        self.assertEqual(labels_row.shape, (n1,))
        self.assertEqual(labels_col.shape, (n2,))

    def test_disconnected(self):
        adjacency = test_graph_disconnect()
        n = adjacency.shape[0]
        labels = self.louvain.fit_transform(adjacency)
        self.assertEqual(len(labels), n)

    def test_options(self):
        adjacency = karate_club()

        # resolution
        louvain = Louvain(resolution=2)
        labels = louvain.fit_transform(adjacency)
        self.assertEqual(len(set(labels)), 7)

        # tolerance
        louvain = Louvain(resolution=2, tol_aggregation=0.1)
        labels = louvain.fit_transform(adjacency)
        self.assertEqual(len(set(labels)), 12)

        # shuffling
        louvain = Louvain(resolution=2, shuffle_nodes=True, random_state=42)
        labels = louvain.fit_transform(adjacency)
        self.assertEqual(len(set(labels)), 9)

        # aggregate graph
        louvain = Louvain(return_adjacency=True)
        labels = louvain.fit_transform(adjacency)
        n_labels = len(set(labels))
        self.assertEqual(louvain.adjacency_.shape, (n_labels, n_labels))
