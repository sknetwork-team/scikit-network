#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Louvain"""

import unittest

from sknetwork.clustering import PropagationClustering
from sknetwork.data.test_graphs import *
from sknetwork.data import karate_club


class TestPropagationClustering(unittest.TestCase):

    def test_disconnected(self):
        adjacency = test_graph_disconnect()
        n = adjacency.shape[0]
        labels = PropagationClustering().fit_transform(adjacency)
        self.assertEqual(len(labels), n)

    def test_options(self):
        adjacency = karate_club()

        # number of iterations
        propagation = PropagationClustering(n_iter=-1)
        labels = propagation.fit_transform(adjacency)
        self.assertEqual(len(set(labels)), 2)

        # aggregate graph
        propagation = PropagationClustering(return_aggregate=True)
        labels = propagation.fit_transform(adjacency)
        n_labels = len(set(labels))
        self.assertEqual(propagation.adjacency_.shape, (n_labels, n_labels))
