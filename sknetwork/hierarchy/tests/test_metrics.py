#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
"""

import unittest
from sknetwork.hierarchy import Paris, tree_sampling_divergence, dasgupta_cost
from sknetwork.toy_graphs import karate_club


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.paris = Paris()
        self.karate_club = karate_club()

    def test_undirected(self):
        adjacency = self.karate_club
        self.paris.fit(adjacency)
        dendrogram = self.paris.dendrogram_
        tsd = tree_sampling_divergence(adjacency, dendrogram, normalized=True)
        self.assertAlmostEqual(tsd, .65, 2)
        dc = dasgupta_cost(adjacency, dendrogram, normalized=True)
        self.assertAlmostEqual(dc, .33, 2)


