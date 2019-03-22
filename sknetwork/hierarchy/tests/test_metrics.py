#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
"""

import unittest
from sknetwork.hierarchy import Paris, tree_sampling_divergence, dasgupta_cost
from sknetwork.toy_graphs import karate_club_graph


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.paris = Paris()
        self.karate_club_graph = karate_club_graph()

    def test_karate_club_graph(self):
        adjacency = self.karate_club_graph
        dendrogram = self.paris.fit(adjacency).dendrogram_
        tsd = tree_sampling_divergence(adjacency, dendrogram, normalized=True)
        self.assertAlmostEqual(tsd, .65, 2)
        dc = dasgupta_cost(adjacency, dendrogram, normalized=True)
        self.assertAlmostEqual(dc, .33, 2)


