#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
"""

import unittest

from sknetwork.hierarchy import Paris, tree_sampling_divergence, dasgupta_score
from sknetwork.data.basic import *


# noinspection PyMissingOrEmptyDocstring
class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.paris = Paris(engine='python')

    def test_undirected(self):
        adjacency = Small().adjacency
        dendrogram = self.paris.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram), 0.697, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram), 0.605, 2)

    def test_directed(self):
        adjacency = DiSmall().adjacency
        dendrogram = self.paris.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram), 0.711, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram), 0.613, 2)

    def test_disconnected(self):
        adjacency = SmallDisconnected().adjacency
        dendrogram = self.paris.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram), 0.752, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram), 0.627, 2)

    def test_options(self):
        adjacency = Small().adjacency
        dendrogram = self.paris.fit_transform(adjacency)
        self.assertAlmostEqual(dasgupta_score(adjacency, dendrogram, weights='degree'), 0.698, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram, weights='uniform'), 0.436, 2)
        self.assertAlmostEqual(tree_sampling_divergence(adjacency, dendrogram, normalized=False), 0.894, 2)
