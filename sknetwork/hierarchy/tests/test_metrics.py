#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
"""

import unittest

from sknetwork.hierarchy import Paris, LouvainHierarchy, dasgupta_cost, dasgupta_score, tree_sampling_divergence
from sknetwork.data.test_graphs import *


# noinspection PyMissingOrEmptyDocstring
class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.paris = Paris()
        self.louvain_hierarchy = LouvainHierarchy()

    def test_undirected(self):
        adjacency = test_graph()
        dendrogram = self.paris.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_cost(adjacency, dendrogram), 3.25, 2)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram), 0.675, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram), 0.533, 2)
        dendrogram = self.louvain_hierarchy.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_cost(adjacency, dendrogram), 4.08, 2)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram), 0.592, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram), 0.485, 2)

    def test_directed(self):
        adjacency = test_digraph()
        dendrogram = self.paris.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram), 0.672, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram), 0.484, 2)
        dendrogram = self.louvain_hierarchy.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram), 0.627, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram), 0.463, 2)

    def test_disconnected(self):
        adjacency = test_graph_disconnect()
        dendrogram = self.paris.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram), 0.752, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram), 0.627, 2)
        dendrogram = self.louvain_hierarchy.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram), 0.691, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram), 0.549, 2)

    def test_options(self):
        adjacency = test_graph()
        dendrogram = self.paris.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram, weights='degree'), 0.659, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram, weights='uniform'), 0.418, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram, normalized=False), 0.738, 2)
