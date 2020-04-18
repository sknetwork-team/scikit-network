#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Louvain"""

import unittest

from sknetwork.clustering import Louvain, BiLouvain, modularity
from sknetwork.data.test_graphs import *
from sknetwork.data import karate_club


class TestLouvainClustering(unittest.TestCase):

    def test_disconnected(self):
        adjacency = test_graph_disconnect()
        n = adjacency.shape[0]
        labels = Louvain().fit_transform(adjacency)
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
        louvain = Louvain(return_aggregate=True)
        labels = louvain.fit_transform(adjacency)
        n_labels = len(set(labels))
        self.assertEqual(louvain.adjacency_.shape, (n_labels, n_labels))

        # aggregate graph
        Louvain(n_aggregations=1, sort_clusters=False).fit(adjacency)
