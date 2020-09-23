#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for betweenness.py"""

import unittest
import numpy as np

from sknetwork.ranking.betweenness import Betweenness
from sknetwork.data.test_graphs import *
from sknetwork.data.toy_graphs import bow_tie


class TestBetweenness(unittest.TestCase):

    def test_bowtie(self):
        adjacency = bow_tie()
        betweenness = Betweenness()
        calc_values = betweenness.fit_transform(adjacency)
        truth_values = np.array([ 4., 0., 0., 0., 0. ])
        self.assertAlmostEqual(np.linalg.norm(truth_values - calc_values), 0)

    def test_disconnected(self):
        adjacency = test_graph_disconnect()
        betweenness = Betweenness()
        with self.assertRaises(ValueError):
            betweenness.fit(adjacency)
