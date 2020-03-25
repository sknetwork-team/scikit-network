#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
"""

import unittest

import numpy as np

from sknetwork.hierarchy import Paris, tree_sampling_divergence, dasgupta_score
from sknetwork.data import karate_club, painters


# noinspection PyMissingOrEmptyDocstring
class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.paris = Paris(engine='python')

    def test_undirected(self):
        adjacency = karate_club()
        dendrogram = self.paris.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram), .666, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram), .487, 2)

    def test_directed(self):
        adjacency = painters()
        dendrogram = self.paris.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram), .584, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram), .468, 2)

    def test_disconnected(self):
        adjacency = np.eye(10)
        dendrogram = self.paris.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram), 1, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram), 0, 2)

    def test_options(self):
        adjacency = karate_club()
        dendrogram = self.paris.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram, weights='degree'), .656, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram, weights='uniform'), .326, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram, normalized=False), .717, 2)
