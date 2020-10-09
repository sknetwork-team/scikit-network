#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for betweenness.py"""

import unittest

from sknetwork.ranking.betweenness import Betweenness
from sknetwork.data.test_graphs import *
from sknetwork.data.toy_graphs import bow_tie


class TestBetweenness(unittest.TestCase):

    def test_basic(self):
        adjacency = test_graph()
        betweenness = Betweenness()
        scores = betweenness.fit_transform(adjacency)
        self.assertEqual(len(scores), adjacency.shape[0])

    def test_bowtie(self):
        adjacency = bow_tie()
        betweenness = Betweenness()
        scores = betweenness.fit_transform(adjacency)
        self.assertEqual(np.sum(scores > 0), 1)

    def test_disconnected(self):
        adjacency = test_graph_disconnect()
        betweenness = Betweenness()
        with self.assertRaises(ValueError):
            betweenness.fit(adjacency)
