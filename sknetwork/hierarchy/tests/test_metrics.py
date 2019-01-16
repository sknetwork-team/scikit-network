# -*- coding: utf-8 -*-
# test for metrics.py
#
# Copyright 2018 Scikit-network Developers.
# Copyright 2018 Bertrand Charpentier <bertrand.charpentier@live.fr>
#
# This file is part of Scikit-network.
#

import unittest
from sknetwork.hierarchy.paris import Paris
from sknetwork.toy_graphs.graph_data import GraphConstants
from sknetwork.hierarchy.metrics import relative_entropy

class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.paris = Paris()
        self.karate_club = GraphConstants.karate_club_graph()

    def test_karate_club_graph(self):
        graph = self.karate_club
        dendrogram = self.paris.fit(graph).dendrogram_
        relative_entropy(graph, dendrogram)
