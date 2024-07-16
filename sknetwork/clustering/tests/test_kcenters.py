#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for KCenters"""
import unittest

from sknetwork.clustering import KCenters
from sknetwork.data.test_graphs import *


class TestKCentersClustering(unittest.TestCase):

    def test_kcenters(self):
        # Test undirected graph
        n_clusters = 2
        adjacency = test_graph()
        n_row = adjacency.shape[0]
        kcenters = KCenters(n_clusters=n_clusters)
        labels = kcenters.fit_predict(adjacency)
        self.assertEqual(len(labels), n_row)
        self.assertEqual(len(set(labels)), n_clusters)

        # Test directed graph
        n_clusters = 3
        adjacency = test_digraph()
        n_row = adjacency.shape[0]
        kcenters = KCenters(n_clusters=n_clusters, directed=True)
        labels = kcenters.fit_predict(adjacency)
        self.assertEqual(len(labels), n_row)
        self.assertEqual(len(set(labels)), n_clusters)

        # Test bipartite graph
        n_clusters = 2
        biadjacency = test_bigraph()
        n_row, n_col = biadjacency.shape
        kcenters = KCenters(n_clusters=n_clusters)
        kcenters.fit(biadjacency)
        labels = kcenters.labels_
        self.assertEqual(len(kcenters.labels_row_), n_row)
        self.assertEqual(len(kcenters.labels_col_), n_col)
        self.assertEqual(len(set(labels)), n_clusters)

    def test_kcenters_error(self):
        # Test value errors
        adjacency = test_graph()
        biadjacency = test_bigraph()

        # test n_clusters error
        kcenters = KCenters(n_clusters=1)
        with self.assertRaises(ValueError):
            kcenters.fit(adjacency)

        # test n_init error
        kcenters = KCenters(n_clusters=2, n_init=0)
        with self.assertRaises(ValueError):
            kcenters.fit(adjacency)

        # test center_position error
        kcenters = KCenters(n_clusters=2, center_position="other")
        with self.assertRaises(ValueError):
            kcenters.fit(biadjacency)
