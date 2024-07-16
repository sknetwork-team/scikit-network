#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
"""

import unittest

from sknetwork.data.test_graphs import *
from sknetwork.data import cyclic_graph
from sknetwork.hierarchy import Paris, LouvainIteration, dasgupta_cost, dasgupta_score, tree_sampling_divergence


# noinspection PyMissingOrEmptyDocstring
class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.paris = Paris()
        self.louvain_iteration = LouvainIteration()

    def test_undirected(self):
        adjacency = cyclic_graph(3)
        dendrogram = self.paris.fit_predict(adjacency)
        self.assertAlmostEqual(dasgupta_cost(adjacency, dendrogram), 2.666, 2)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram), 0.111, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram), 0.0632, 3)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram, normalized=False), 0.0256, 3)
        adjacency = test_graph()
        dendrogram = self.paris.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_cost(adjacency, dendrogram), 4.26, 2)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram), 0.573, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram), 0.304, 2)
        dendrogram = self.louvain_iteration.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_cost(adjacency, dendrogram), 4.43, 2)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram), 0.555, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram), 0.286, 2)

    def test_directed(self):
        adjacency = test_digraph()
        dendrogram = self.paris.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram), 0.566, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram), 0.318, 2)
        dendrogram = self.louvain_iteration.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram), 0.55, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram), 0.313, 2)

    def test_disconnected(self):
        adjacency = test_disconnected_graph()
        dendrogram = self.paris.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram), 0.682, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram), 0.464, 2)
        dendrogram = self.louvain_iteration.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram), 0.670, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram), 0.594, 2)

    def test_options(self):
        adjacency = test_graph()
        dendrogram = self.paris.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram, weights='degree'), 0.573, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram, weights='uniform'), 0.271, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram, normalized=False), 0.367, 2)
