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


