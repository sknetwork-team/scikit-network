#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
"""

import unittest

from sknetwork.hierarchy import Paris, dasgupta_cost, dasgupta_score, tree_sampling_divergence
from sknetwork.data.test_graphs import *


# noinspection PyMissingOrEmptyDocstring
class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.paris = Paris()

    def test_undirected(self):
        adjacency = test_graph()
        dendrogram = self.paris.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_cost(adjacency, dendrogram), 3.25, 2)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram), 0.675, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram), 0.533, 2)

    def test_directed(self):
        adjacency = test_digraph()
        dendrogram = self.paris.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram), 0.672, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram), 0.484, 2)

    def test_disconnected(self):
        adjacency = test_graph_disconnect()
        dendrogram = self.paris.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram), 0.752, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram), 0.627, 2)

    def test_options(self):
        adjacency = test_graph()
        dendrogram = self.paris.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram, weights='degree'), 0.659, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram, weights='uniform'), 0.418, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram, normalized=False), 0.738, 2)
