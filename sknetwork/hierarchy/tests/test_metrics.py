#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
"""

import unittest
from sknetwork.hierarchy.paris import Paris
from sknetwork.toy_graphs.graph_data import karate_club_graph
from sknetwork.hierarchy.metrics import tree_sampling_divergence, dasgupta_cost


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.paris = Paris()
        self.karate_club = karate_club_graph()

    def test_karate_club_graph(self):
        adj_matrix = self.karate_club
        dendrogram = self.paris.fit(adj_matrix).dendrogram_
        tsd = tree_sampling_divergence(adj_matrix, dendrogram, normalized=True)
        self.assertLessEqual(tsd, .7)
        self.assertGreaterEqual(tsd, .6)
        dc = dasgupta_cost(adj_matrix, dendrogram, normalized=True)
        self.assertLessEqual(dc, .4)
        self.assertGreaterEqual(dc, .3)


