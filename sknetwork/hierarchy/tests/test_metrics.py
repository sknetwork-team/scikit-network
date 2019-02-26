# -*- coding: utf-8 -*-
# test for metrics.py
"""tests for hierarchy metrics"""

import unittest
from sknetwork.hierarchy.paris import Paris
from sknetwork.toy_graphs.graph_data import karate_club_graph
from sknetwork.hierarchy.metrics import relative_entropy


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.paris = Paris()
        self.karate_club = karate_club_graph()

    def test_karate_club_graph(self):
        graph = self.karate_club
        dendrogram = self.paris.fit(graph).dendrogram_
        relative_entropy(graph, dendrogram)
