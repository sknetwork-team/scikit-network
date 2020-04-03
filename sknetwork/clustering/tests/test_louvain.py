#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Louvain"""

import unittest

from sknetwork import is_numba_available
from sknetwork.clustering import Louvain, BiLouvain, modularity
from sknetwork.data.test_graphs import *
from sknetwork.data import karate_club


# noinspection PyMissingOrEmptyDocstring
class TestLouvainClustering(unittest.TestCase):

    def setUp(self):
        self.louvain = Louvain(engine='python')
        self.bilouvain = BiLouvain(engine='python')
        if is_numba_available:
            self.louvain_numba = Louvain(engine='numba')
            self.bilouvain_numba = BiLouvain(engine='numba')
        else:
            with self.assertRaises(ValueError):
                Louvain(engine='numba')

    def test_undirected(self):
        adjacency = test_graph()
        labels = self.louvain.fit_transform(adjacency)
        self.assertEqual(labels.shape, (10,))
        self.assertAlmostEqual(modularity(adjacency, labels), 0.503, 2)
        if is_numba_available:
            labels = self.louvain_numba.fit_transform(adjacency)
            self.assertEqual(labels.shape, (10,))
            self.assertAlmostEqual(modularity(adjacency, labels), 0.503, 2)

    def test_directed(self):
        adjacency = test_digraph()
        labels = self.louvain.fit_transform(adjacency)
        self.assertEqual(labels.shape, (10,))
        self.assertAlmostEqual(modularity(adjacency, labels), 0.548, 2)

    def test_bipartite(self):
        biadjacency = test_bigraph()
        n1, n2 = biadjacency.shape
        self.bilouvain.fit(biadjacency)
        labels_row = self.bilouvain.labels_row_
        labels_col = self.bilouvain.labels_col_
        self.assertEqual(labels_row.shape, (n1,))
        self.assertEqual(labels_col.shape, (n2,))
        if is_numba_available:
            self.bilouvain_numba.fit(biadjacency)
            labels_row = self.bilouvain_numba.labels_row_
            labels_col = self.bilouvain_numba.labels_col_
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
        louvain = Louvain(engine='python', resolution=2)
        labels = louvain.fit_transform(adjacency)
        self.assertEqual(len(set(labels)), 7)

        # tolerance
        louvain = Louvain(engine='python', resolution=2, tol_aggregation=0.1)
        labels = louvain.fit_transform(adjacency)
        self.assertEqual(len(set(labels)), 12)

        # shuffling
        louvain = Louvain(engine='python', resolution=2, shuffle_nodes=True, random_state=42)
        labels = louvain.fit_transform(adjacency)
        self.assertEqual(len(set(labels)), 9)

        # aggregate graph
        louvain = Louvain(engine='python', return_graph=True)
        labels = louvain.fit_transform(adjacency)
        n_labels = len(set(labels))
        self.assertEqual(louvain.adjacency_.shape, (n_labels, n_labels))

